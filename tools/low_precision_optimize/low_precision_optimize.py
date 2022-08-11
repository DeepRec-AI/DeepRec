import os
import shutil
import sys
import tempfile

import numpy as np
import tensorflow as tf
import tf_graph_transform_utils as util
from calibrate import non_linear_quant_params_search
from simple_graph import SimpleGraph, SimpleNode
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import meta_graph, ops, versions
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.util import quantize_embedding_variable

INT8 = 'INT8'
BF16 = 'BF16'
FP16 = 'FP16'

ev_attrs = [
    '_block_num',
    '_counter_type',
    '_default_value_dim',
    '_emb_index',
    '_false_positive_probability',
    '_filter_freq',
    '_ht_partition_num',
    '_ht_type',
    '_init_data_source',
    '_invalid_key_type',
    '_is_sparse',
    '_l2_weight_threshold',
    '_layout',
    '_max_element_size',
    '_save_slice_info',
    '_slot_index',
    '_slot_num',
    '_steps_to_live',
    '_storage_path',
    '_storage_size',
    '_storage_type',
]


def _ts(name):
    return util.get_canonical_input_name(name)


def _nd(name):
    return util.get_node_name_from_input(name)


def get_all_variables():
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    all_variables += tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    return list(set(all_variables))


def get_variable_by_name(name):
    all_variables = get_all_variables()
    for var in all_variables:
        if var.name == name:
            return var
    return None


def update_op_inputs(graph, rename_dict):
    rename_dict = {_ts(k): v for k, v in rename_dict.items()}
    src_names = list(rename_dict.keys())
    for op in graph.get_operations():
        for i, tensor in enumerate(op.inputs):
            if tensor.name in src_names:
                op._update_input(i, rename_dict[tensor.name])


def create_new_meta_graph_def(graph, meta_graph_def, new_graph_def, new_saver=None):
    new_nodes = [node.name for node in new_graph_def.node]
    old_nodes = [node.name for node in graph.as_graph_def().node]
    removed_nodes = list(set(old_nodes).difference(set(new_nodes)))
    new_mgd = meta_graph_pb2.MetaGraphDef()
    # Add meta info def
    meta_info_def = meta_graph_pb2.MetaGraphDef.MetaInfoDef()
    meta_info_def.tensorflow_version = versions.__version__
    meta_info_def.tensorflow_git_version = versions.__git_version__
    stripped_op_list = meta_graph.stripped_op_list_for_graph(new_graph_def)
    meta_info_def.stripped_op_list.MergeFrom(stripped_op_list)
    new_mgd.meta_info_def.MergeFrom(meta_info_def)
    # Add graph def
    new_mgd.graph_def.MergeFrom(new_graph_def)
    # Add saver def
    new_mgd.saver_def.MergeFrom(new_saver.saver_def)
    # Add collection list
    clist = graph.get_all_collection_keys()
    exclude_nodes = [_ts(name) for name in removed_nodes] + removed_nodes
    for ctype in clist:
        if ctype in [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.TRAINABLE_VARIABLES]:
            meta_graph.add_collection_def(
                new_mgd,
                ctype,
                graph=graph,
                export_scope=None,
                exclude_nodes=exclude_nodes,
            )
            values = new_mgd.collection_def[ctype].bytes_list.value
            new_mgd.collection_def[ctype].bytes_list.Clear()
            for value in values:
                proto = ops.get_collection_proto_type(ctype)()
                proto.ParseFromString(value)
                if proto.variable_name in exclude_nodes:
                    continue
                for attr in ['initial_value_name', 'snapshot_name', 'initializer_name']:
                    if getattr(proto, attr) in exclude_nodes:
                        setattr(proto, attr, proto.variable_name)
                new_value = proto.SerializeToString()
                new_mgd.collection_def[ctype].bytes_list.value.append(new_value)

    for tag in meta_graph_def.meta_info_def.tags:
        new_mgd.meta_info_def.tags.append(tag)
    for key in meta_graph_def.signature_def:
        new_mgd.signature_def[key].CopyFrom(meta_graph_def.signature_def[key])
    return new_mgd


def remove_redundant_quants(session, graph_def):
    simple_graph = SimpleGraph(graph_def)

    def _get_pattern():
        pl = list()
        pl.append(SimpleNode('quantize', 'QuantizeV2', ['dequantize', '0', '1'], ['0']))
        pl.append(SimpleNode('dequantize', 'Dequantize', ['2', '3', '4'], ['quantize']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    pattern, first_key = _get_pattern()
    ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
    for ptm in ptm_list:
        dequantize_op = session.graph.get_operation_by_name(ptm['dequantize'])
        update_op_inputs(session.graph, {ptm['quantize']: dequantize_op.inputs[0]})


def remove_redundant_casts(session, graph_def):
    simple_graph = SimpleGraph(graph_def)

    def _get_pattern():
        pl = list()
        pl.append(SimpleNode('cast2src', 'Cast', ['cast2dst'], ['0']))
        pl.append(SimpleNode('cast2dst', 'Cast', ['0'], ['cast2src']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    pattern, first_key = _get_pattern()
    ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
    for ptm in ptm_list:
        cast2src_op = session.graph.get_operation_by_name(ptm['cast2src'])
        cast2dst_op = session.graph.get_operation_by_name(ptm['cast2dst'])
        if cast2dst_op.get_attr('SrcT') == cast2src_op.get_attr('DstT'):
            update_op_inputs(session.graph, {ptm['cast2src']: cast2dst_op.inputs[0]})


def dense_opt(session, graph_def, opt_config, data_type, calib_file):
    simple_graph = SimpleGraph(graph_def)
    update_dict = dict()
    calib_data = None
    if calib_file:
        calib_data = np.load(calib_file, allow_pickle=True, encoding='bytes')

    def _calibrate(ts_name):
        assert calib_data is not None, 'Calibration data needed for INT8 optimization.'
        values = [session.run(ts_name, feed_dict=fd) for fd in calib_data]
        values = np.concatenate([v.ravel() for v in values])
        return non_linear_quant_params_search(values)

    def _get_matmul_pattern(with_bias, with_relu):
        pl = list()
        output = ['0']
        if with_relu:
            tmp_input = ['bias_add'] if with_bias else ['matmul']
            pl.append(SimpleNode('relu', 'Relu', tmp_input, ['1']))
            output = ['relu']
        if with_bias:
            pl.append(SimpleNode('bias_add', 'BiasAdd', ['matmul', '2'], output))
            output = ['bias_add']
        pl.append(SimpleNode('matmul', 'MatMul', ['0', '1'], output))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    def _get_weight_data(node_name, input_index=1):
        weight_name = util.get_input_target_op_name(
            simple_graph, node_name, input_index, 'Const', {'Identity': [0]}
        )
        if weight_name:
            data = util.get_const_value_by_name(graph_def, weight_name, simple_graph)
        else:
            node = util.get_node_by_name(graph_def, simple_graph, node_name)
            try:
                data = session.run(_ts(node.input[input_index]))
            except Exception:
                return None
        return data

    def _optimize(with_bias, with_relu):
        pattern, first_key = _get_matmul_pattern(with_bias, with_relu)
        ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
        ptm_list = [ptm for ptm in ptm_list if ptm['matmul'] not in update_dict]
        if opt_config:
            ptm_list = [ptm for ptm in ptm_list if ptm['matmul'] in opt_config]
        for ptm in ptm_list:
            if ptm['matmul'] in update_dict:
                continue
            w_data = _get_weight_data(ptm['matmul'])
            if with_bias:
                bias_data = _get_weight_data(ptm['bias_add'])
            if w_data is None or (with_bias and bias_data is None):
                continue
            node = util.get_node_by_name(graph_def, simple_graph, ptm['matmul'])
            opt_dtype = opt_config.get(node.name) if opt_config else data_type
            print(f'Optimize dense op to {opt_dtype}: {node.name}')
            update_dict[node.name] = [opt_dtype]
            pref = node.name
            dense_op = session.graph.get_operation_by_name(node.name)
            if opt_dtype in [BF16, FP16]:
                tf_dtype = tf.bfloat16 if opt_dtype == BF16 else tf.float16
                w_f16_ts = tf.constant(
                    value=tf.cast(w_data, tf_dtype).eval(),
                    dtype=tf_dtype,
                    name=f'{pref}/{opt_dtype.lower()}_weight',
                )
                in_f16_ts = tf.cast(
                    dense_op.inputs[0], tf_dtype, name=f'{pref}/{opt_dtype.lower()}'
                )
                out_f16_ts = tf.matmul(
                    a=in_f16_ts,
                    b=w_f16_ts,
                    transpose_a=dense_op.get_attr('transpose_a'),
                    transpose_b=dense_op.get_attr('transpose_b'),
                    name=f'{pref}/{opt_dtype.lower()}_matmul',
                )
                if with_bias:
                    bias_f16_ts = tf.constant(
                        value=tf.cast(bias_data, tf_dtype).eval(),
                        dtype=tf_dtype,
                        name=f'{pref}/{opt_dtype.lower()}_bias',
                    )
                    out_f16_ts = tf.nn.bias_add(out_f16_ts, bias_f16_ts)
                out_f16_ts = tf.nn.relu(out_f16_ts) if with_relu else out_f16_ts
                out_fp32_ts = tf.cast(out_f16_ts, tf.float32, name=f'{pref}/fp32')
                update_op_inputs(session.graph, {ptm[first_key]: out_fp32_ts})
                continue
            elif opt_dtype != INT8:
                raise Exception(f'Unsupported data type: {opt_dtype}')
            # Optimize to INT8
            # Update weight
            w_max_abs_val = np.max(np.abs(w_data))
            w_scale = np.array(w_max_abs_val / 127.0)
            w_int8_data = np.int8(np.round(w_data / w_scale))
            w_min_ts = tf.constant(-1 * w_max_abs_val, tf.float32, name=f'{pref}/w_min')
            w_max_ts = tf.constant(w_max_abs_val, tf.float32, name=f'{pref}/w_max')
            w_int8_ts = tf.constant(w_int8_data, tf.qint8, name=f'{pref}/int8_weight')
            # Update input
            in_min_val, in_max_val = _calibrate(_ts(node.input[0]))
            in_min_ts = tf.constant(in_min_val, tf.float32, name=f'{pref}/in_min')
            in_max_ts = tf.constant(in_max_val, tf.float32, name=f'{pref}/in_max')
            in_int8_ts, _, _ = tf.raw_ops.QuantizeV2(
                input=dense_op.inputs[0],
                min_range=in_min_ts,
                max_range=in_max_ts,
                T=tf.quint8,
                mode='MIN_FIRST',
                name=f'{pref}/int8_input',
            )
            # Add requantize scale
            out_min_val, out_max_val = _calibrate(_ts(ptm[first_key]))
            req_min_val, req_max_val = 0.0, (out_max_val - out_min_val) * 256.0 / 255.0
            req_min_ts = tf.constant(req_min_val, tf.float32, name=f'{pref}/req_min')
            req_max_ts = tf.constant(req_max_val, tf.float32, name=f'{pref}/req_max')
            # Update bias
            in_scale = (in_max_val - in_min_val) / 255.0
            in_zero_point = -1.0 * round(in_min_val / in_scale)
            compensation = np.sum(-1.0 * in_zero_point * w_int8_data, 0)
            out_scale = (out_max_val - out_min_val) / 255.0
            out_zero_point = -1.0 * round(out_min_val / out_scale)
            deq_scale = w_scale * in_scale
            compensation += out_zero_point * out_scale / deq_scale
            if with_bias:
                bias_int_data = np.int32(compensation + bias_data / deq_scale)
            else:
                bias_int_data = np.int32(compensation)
            bias_int_ts = tf.constant(bias_int_data, tf.qint32, name=f'{pref}/int_bias')
            # Update MatMul
            if with_relu:
                quant_matmul = tf.raw_ops.QuantizedMatMulWithBiasAndReluAndRequantize
            else:
                quant_matmul = tf.raw_ops.QuantizedMatMulWithBiasAndRequantize
            matmul_int8_ts, _, _ = quant_matmul(
                a=in_int8_ts,
                b=w_int8_ts,
                bias=bias_int_ts,
                min_a=in_min_ts,
                max_a=in_max_ts,
                min_b=w_min_ts,
                max_b=w_max_ts,
                min_freezed_output=req_min_ts,
                max_freezed_output=req_max_ts,
                Toutput=tf.quint8,
                input_quant_mode='MIN_FIRST',
                name=f'{pref}/int8_matmul',
            )
            # Add dequantize
            out_min_ts = tf.constant(out_min_val, tf.float32, name=f'{pref}/out_min')
            out_max_ts = tf.constant(out_max_val, tf.float32, name=f'{pref}/out_max')
            out_fp32_ts = tf.raw_ops.Dequantize(
                input=matmul_int8_ts,
                min_range=out_min_ts,
                max_range=out_max_ts,
                mode='MIN_FIRST',
                name=f'{pref}/dequantize',
            )
            update_op_inputs(session.graph, {ptm[first_key]: out_fp32_ts})

    for with_relu in [True, False]:
        for with_bias in [True, False]:
            _optimize(with_bias, with_relu)

    remove_redundant_quants(session, session.graph.as_graph_def())
    remove_redundant_casts(session, session.graph.as_graph_def())

    return update_dict


def update_embedding_vars(session):
    update_dict = dict()
    node_dic = {nd.name: nd for nd in session.graph_def.node}
    for op in session.graph.get_operations():
        if op.type == 'KvResourceImportV2':
            var = get_variable_by_name(op.inputs[1].name)
            # print(f'Update embedding variable: {var.name}')
            var._initial_value = op.inputs[3]
            name = _nd(op.inputs[5].name)
            var._invalid_key = util.get_const_value(node_dic[name])
            update_dict[var.name] = op.name

    return update_dict


def embedding_opt(session, graph_def, opt_config, data_type):
    simple_graph = SimpleGraph(graph_def)

    def _get_gather_pattern():
        pl = list()
        pl.append(SimpleNode('gather', 'GatherV2', ['read', '0', '1'], ['0']))
        pl.append(SimpleNode('read', 'Identity', ['embed'], ['gather']))
        pl.append(SimpleNode('embed', 'Const', [], ['read']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    update_dict = dict()
    pattern, first_key = _get_gather_pattern()
    ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
    if opt_config:
        ptm_list = [ptm for ptm in ptm_list if ptm['embed'] in opt_config]
    for ptm in ptm_list:
        embed_node = util.get_node_by_name(graph_def, simple_graph, ptm['embed'])
        opt_dtype = opt_config.get(embed_node.name) if opt_config else data_type
        if embed_node.name not in update_dict:
            print(f'Optimize embedding to {opt_dtype}: {embed_node.name}')
            # Add variables
            fp32_data = util.get_const_value_by_name(
                graph_def, ptm['embed'], simple_graph
            )
            if opt_dtype == INT8:
                max_abs_val = np.max(np.abs(fp32_data), axis=0)
                scale = np.array(max_abs_val / 127.0)
                int8_data = np.round(fp32_data / scale)
                int8_name = f'{embed_node.name}/int8_data'
                int8_var = tf.get_variable(int8_name, fp32_data.shape, tf.int8)
                session.run(int8_var.assign(int8_data))
                scale_name = f'{embed_node.name}/int8_scale'
                scale_var = tf.get_variable(scale_name, scale.shape, tf.float32)
                session.run(scale_var.assign(scale))
                update_dict[embed_node.name] = [int8_var, scale_var, opt_dtype]
            elif opt_dtype in [BF16, FP16]:
                tf_dtype = tf.bfloat16 if opt_dtype == BF16 else tf.float16
                f16_name = f'{embed_node.name}/{opt_dtype.lower()}_data'
                f16_var = tf.get_variable(f16_name, fp32_data.shape, tf_dtype)
                session.run(f16_var.assign(tf.cast(fp32_data, tf_dtype)))
                update_dict[embed_node.name] = [f16_var, opt_dtype]
            else:
                raise Exception(f'Unsupported data type: {opt_dtype}')
        # Update Graph
        gather_op = session.graph.get_operation_by_name(ptm['gather'])
        opt_gather = tf.gather(
            params=update_dict[embed_node.name][0],
            indices=gather_op.inputs[1],
            axis=gather_op.inputs[2],
            batch_dims=gather_op.get_attr('batch_dims'),
            name=f'{ptm["gather"]}/{opt_dtype.lower()}',
        )
        cast_name = f'{ptm["gather"]}/cast_to_fp32'
        update_tensor = tf.cast(opt_gather, dtype=tf.float32, name=cast_name)
        if opt_dtype == INT8:
            rescale_name = f'{ptm["gather"]}/rescale'
            scale_var = update_dict[embed_node.name][1]
            update_tensor = tf.multiply(update_tensor, scale_var, name=rescale_name)
        update_op_inputs(session.graph, {ptm['gather']: update_tensor})

    return update_dict


def embedding_var_opt(session, graph_def, opt_config, data_type, variable_path):
    simple_graph = SimpleGraph(graph_def)

    def _get_gather_pattern():
        pl = list()
        pl.append(SimpleNode('gather', 'KvResourceGather', ['embed', '0', '1'], ['0']))
        pl.append(SimpleNode('embed', 'KvVarHandleOp', [], ['gather']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    def _get_tf_dtype(dtype):
        if dtype == INT8:
            return tf.int8
        elif dtype in [BF16, FP16]:
            return tf.bfloat16 if dtype == BF16 else tf.float16
        else:
            raise Exception(f'Unsupported data type: {dtype}')

    update_dict = dict()
    pattern, first_key = _get_gather_pattern()
    ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
    if opt_config:
        ptm_list = [ptm for ptm in ptm_list if ptm['embed'] in opt_config]
    for ptm in ptm_list:
        embed_name = ptm['embed']
        opt_dtype = opt_config.get(embed_name) if opt_config else data_type
        if embed_name not in update_dict:
            embed_node = util.get_simple_node_by_name(simple_graph, embed_name)
            if embed_node.output_nodes != [ptm['gather']]:
                continue
            print(f'Optimize embedding variable to {opt_dtype}: {embed_name}')
            # Add variables
            var = get_variable_by_name(_ts(embed_name))
            embedding_dim = var.shape[0].value
            value_dtype = _get_tf_dtype(opt_dtype)
            opt_var = tf.get_embedding_variable(
                name=f'{embed_name}/{opt_dtype.lower()}_data',
                embedding_dim=embedding_dim,
                key_dtype=var._invalid_key_type,
                value_dtype=value_dtype,
                initializer=tf.zeros_initializer(value_dtype),
            )
            for attr in ev_attrs:
                setattr(opt_var, attr, getattr(var, attr))
            if opt_dtype == INT8:
                scale_name = f'{embed_name}/int8_scale'
                scale_var = tf.get_variable(scale_name, [embedding_dim], tf.float32)
                update_dict[embed_name] = [var, opt_var, scale_var, opt_dtype]
            elif opt_dtype in [BF16, FP16]:
                update_dict[embed_name] = [var, opt_var, opt_dtype]

        # Update Graph
        gather_op = session.graph.get_operation_by_name(ptm['gather'])
        var, opt_var = update_dict[embed_name][0], update_dict[embed_name][1]
        use_default = gather_op.get_attr('is_use_default_value_tensor')
        if use_default:
            init_val = gather_op.inputs[2]
            if opt_dtype == INT8:
                init_val = tf.divide(init_val, update_dict[embed_name][2])
                init_val = tf.raw_ops.ClipByValue(init_val, -128, 127)
            init_val = tf.cast(init_val, value_dtype)
        else:
            init_val = tf.constant(1, dtype=value_dtype)
        opt_gather = gen_kv_variable_ops.kv_resource_gather(
            resource=opt_var._handle,
            indices=gather_op.inputs[1],
            default_value=init_val,
            is_use_default_value_tensor=use_default,
            name=f'{ptm["gather"]}/{opt_dtype.lower()}',
        )
        cast_name = f'{ptm["gather"]}/cast_to_fp32'
        update_tensor = tf.cast(opt_gather, dtype=tf.float32, name=cast_name)
        if opt_dtype == INT8:
            rescale_name = f'{ptm["gather"]}/rescale'
            scale_var = update_dict[embed_name][2]
            update_tensor = tf.multiply(update_tensor, scale_var, name=rescale_name)
        update_op_inputs(session.graph, {ptm['gather']: update_tensor})

    if len(update_dict) == 0:
        return update_dict, None

    # Convert checkpoint
    input_path = variable_path
    for opt_dtype in [INT8, BF16, FP16]:
        opt_dict = {k: v for k, v in update_dict.items() if v[-1] == opt_dtype}
        if len(opt_dict) > 0:
            names = [_nd(key) for key in opt_dict.keys()]
            quant_names = [_nd(opt_dict[name][1].name) for name in names]
            if opt_dtype == INT8:
                scale_names = [_nd(opt_dict[name][2].name) for name in names]
                scale_variables = [opt_dict[name][2] for name in names]
                session.run(tf.variables_initializer(scale_variables))
            else:
                scale_names = []
            tmp_path = tempfile.mkdtemp(dir='.')
            opt_path = f'{tmp_path}/variables'
            tf_dtype = _get_tf_dtype(opt_dtype)
            quantize_embedding_variable.quantize_by_name(
                variable_path, opt_path, names, quant_names, scale_names, tf_dtype
            )
            if variable_path != input_path:
                shutil.rmtree(variable_path)
            variable_path = opt_path
    return update_dict, opt_path


def optimize(model_path, save_path, opt_config=None, data_type=BF16, calib_file=None):
    saved_model = loader_impl._parse_saved_model(model_path)
    tags = saved_model.meta_graphs[0].meta_info_def.tags
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, tags, model_path)
        signature_keys = list(meta_graph_def.signature_def.keys())
        signature_def = meta_graph_def.signature_def[signature_keys[0]]
        model_outputs = [_nd(v.name) for v in signature_def.outputs.values()]
        init_op = loader_impl.get_init_op(meta_graph_def)
        if init_op is not None:
            model_outputs.append(init_op.name)
        frozen_gdef = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, model_outputs
        )

        # Embedding & Dense optimization
        dense_opt_dict = dense_opt(sess, frozen_gdef, opt_config, data_type, calib_file)
        embed_opt_dict = embedding_opt(sess, frozen_gdef, opt_config, data_type)
        ev_dict = update_embedding_vars(sess)
        if len(ev_dict) > 0:
            model_outputs.append(_nd(tf.train.get_global_step().name))

        def _extract_sub_graph(outputs):
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            util.remove_underscore_class_attr(graph_def)
            return tf.graph_util.extract_sub_graph(graph_def, outputs)

        def _save(save_path):
            sub_graph_def = _extract_sub_graph(model_outputs)
            node_names = [node.name for node in sub_graph_def.node]
            variables = [v for v in get_all_variables() if _nd(v.name) in node_names]
            init_name = tf.variables_initializer(variables).name
            saver = tf.train.Saver(variables, sharded=True, allow_empty=True)
            saver.save(sess, save_path, write_meta_graph=False, write_state=False)
            return saver, init_name

        # Create Saver
        tmp_path = tempfile.mkdtemp(dir='.')
        variable_path = f'{tmp_path}/variables'
        saver, init_name = _save(variable_path)
        # Optimize embedding variables
        ev_opt_dict, opt_variable_path = embedding_var_opt(
            sess, frozen_gdef, opt_config, data_type, variable_path
        )
        if len(ev_opt_dict) > 0:
            saver, init_name = _save(variable_path)
            variable_path = opt_variable_path

        saver_nodes = [
            saver.saver_def.restore_op_name,
            _nd(saver.saver_def.filename_tensor_name),
            _nd(saver.saver_def.save_tensor_name),
        ]
        graph_def = _extract_sub_graph(model_outputs + saver_nodes + [init_name])
        graph = sess.graph

    # Create new meta graph def
    new_mgd = create_new_meta_graph_def(graph, meta_graph_def, graph_def, saver)

    # Export saved_model
    tf.reset_default_graph()
    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph.import_scoped_meta_graph(new_mgd)
        restore_feed_dict = {new_mgd.saver_def.filename_tensor_name: variable_path}
        sess.run(new_mgd.saver_def.restore_op_name, restore_feed_dict)
        # Update embedding variables
        update_embedding_vars(sess)
        # Update assets file
        assets_collection = None
        asset_dict = loader_impl.get_asset_tensors(model_path, meta_graph_def)
        if asset_dict is not None:
            for tensor_name, filename in asset_dict.items():
                node_name = _nd(tensor_name)
                if node_name in [nd.name for nd in sess.graph_def.node]:
                    asset_op = sess.graph.get_operation_by_name(node_name)
                    ts_proto = TensorProto(
                        tensor_shape=asset_op.get_attr('value').tensor_shape,
                        dtype=asset_op.get_attr('value').dtype,
                        string_val=[filename],
                    )
                    asset_op._set_attr('value', tf.AttrValue(tensor=ts_proto))
                    asset_name = asset_op.outputs[0]
                else:
                    asset_name = tf.constant(filename, name=node_name)
                tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_name)
            assets_collection = tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
        main_op = sess.graph.get_operation_by_name(init_op.name) if init_op else None
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=new_mgd.meta_info_def.tags,
            signature_def_map=new_mgd.signature_def,
            assets_collection=assets_collection,
            main_op=main_op,
        )
        builder.save()
        if len(ev_opt_dict) > 0:
            target_path = f'{save_path}/variables'
            shutil.rmtree(target_path)
            shutil.copytree(os.path.dirname(variable_path), target_path)
            shutil.rmtree(os.path.dirname(variable_path))
        shutil.rmtree(tmp_path)

        print('Optmization Result:')
        for key, value in dense_opt_dict.items():
            print(f'Optimize dense op to {value[-1]}: {key}')
        for key, value in embed_opt_dict.items():
            print(f'Optimize embedding to {value[-1]}: {key}')
        for key, value in ev_opt_dict.items():
            print(f'Optimize embedding variable to {value[-1]}: {key}')


if __name__ == '__main__':
    model_path = sys.argv[1]
    save_path = sys.argv[2]
    data_type = sys.argv[3]
    calib_file = sys.argv[4] if len(sys.argv) > 4 else None
    opt_dict = None
    optimize(model_path, save_path, opt_dict, data_type, calib_file)
