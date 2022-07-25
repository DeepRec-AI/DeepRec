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
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.training import saver as tf_saver


def _ts(name):
    return util.get_canonical_input_name(name)


def _nd(name):
    return util.get_node_name_from_input(name)


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
    calib_data = None
    if calib_file:
        calib_data = np.load(calib_file, allow_pickle=True, encoding='bytes')

    def _calibrate(ts_name):
        assert calib_data is not None, 'Calibration data needed for INT8 optimization.'
        values = [session.run(ts_name, feed_dict=fd) for fd in calib_data]
        values = np.concatenate([v.ravel() for v in values])
        return non_linear_quant_params_search(values)

    simple_graph = SimpleGraph(graph_def)

    def _get_matmul_pattern(with_relu):
        pl = list()
        if with_relu:
            pl.append(SimpleNode('relu', 'Relu', ['bias_add'], ['0']))
        tmp_output = ['relu'] if with_relu else ['1']
        pl.append(SimpleNode('bias_add', 'BiasAdd', ['matmul', 'read_b'], tmp_output))
        pl.append(SimpleNode('matmul', 'MatMul', ['1', 'read_w'], ['bias_add']))
        pl.append(SimpleNode('read_w', 'Identity', ['weight'], ['2']))
        pl.append(SimpleNode('weight', 'Const', [], ['read_w']))
        pl.append(SimpleNode('read_b', 'Identity', ['bias'], ['3']))
        pl.append(SimpleNode('bias', 'Const', [], ['read_b']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    update_dict = dict()
    for with_relu in [True, False]:
        pattern, first_key = _get_matmul_pattern(with_relu)
        ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
        ptm_list = [ptm for ptm in ptm_list if ptm['matmul'] not in update_dict]
        if opt_config:
            ptm_list = [ptm for ptm in ptm_list if ptm['matmul'] in opt_config]
        for ptm in ptm_list:
            node = util.get_node_by_name(graph_def, simple_graph, ptm['matmul'])
            opt_dtype = opt_config.get(node.name) if opt_config else data_type
            if node.name in update_dict:
                continue
            print(f'Optimize dense op to {opt_dtype}: {node.name}')
            update_dict[node.name] = [opt_dtype]
            pref = node.name
            w_data = util.get_const_value_by_name(graph_def, ptm['weight'])
            bias_data = util.get_const_value_by_name(graph_def, ptm['bias'])
            dense_op = session.graph.get_operation_by_name(node.name)
            if opt_dtype == 'BF16':
                w_bf16_ts = tf.constant(
                    value=tf.cast(w_data, tf.bfloat16).eval(),
                    dtype=tf.bfloat16,
                    name=f'{pref}/bf16_weight',
                )
                in_bf16_ts = tf.cast(dense_op.inputs[0], tf.bfloat16)
                matmul_bf16_ts = tf.matmul(
                    a=in_bf16_ts,
                    b=w_bf16_ts,
                    transpose_a=dense_op.get_attr('transpose_a'),
                    transpose_b=dense_op.get_attr('transpose_b'),
                    name=f'{pref}/bf16_matmul',
                )
                bias_bf16_ts = tf.constant(
                    value=tf.cast(bias_data, tf.bfloat16).eval(),
                    dtype=tf.bfloat16,
                    name=f'{pref}/bf16_bias',
                )
                out_bf16_ts = tf.nn.bias_add(matmul_bf16_ts, bias_bf16_ts)
                out_bf16_ts = tf.nn.relu(out_bf16_ts) if with_relu else out_bf16_ts
                out_fp32_ts = tf.cast(out_bf16_ts, tf.float32)
                update_op_inputs(session.graph, {ptm[first_key]: out_fp32_ts})
                continue
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
            bias_int_data = np.int32(compensation + bias_data / deq_scale)
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

    remove_redundant_quants(session, session.graph.as_graph_def())
    remove_redundant_casts(session, session.graph.as_graph_def())

    return update_dict


def embedding_opt(session, graph_def, opt_config, data_type):
    simple_graph = SimpleGraph(graph_def)

    def _get_gather_pattern():
        pl = list()
        pl.append(SimpleNode('gather', 'GatherV2', ['read', '0', '1'], ['0']))
        pl.append(SimpleNode('read', 'Identity', ['embed'], ['1']))
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
            # Add variables
            print(f'Optimize embedding to {opt_dtype}: {embed_node.name}')
            fp32_data = util.get_const_value_by_name(graph_def, ptm['embed'])
            if opt_dtype == 'INT8':
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
            elif opt_dtype == 'BF16':
                bf16_name = f'{embed_node.name}/bf16_data'
                bf16_var = tf.get_variable(bf16_name, fp32_data.shape, tf.bfloat16)
                session.run(bf16_var.assign(tf.cast(fp32_data, tf.bfloat16)))
                update_dict[embed_node.name] = [bf16_var, opt_dtype]
        # Update Graph
        gather_op = session.graph.get_operation_by_name(ptm['gather'])
        opt_gather = tf.gather(
            params=update_dict[embed_node.name][0],
            indices=gather_op.inputs[1],
            axis=gather_op.inputs[2],
            batch_dims=gather_op.get_attr('batch_dims'),
            name=f'{ptm["gather"]}/{opt_dtype}',
        )
        cast_name = f'{ptm["gather"]}/cast_to_fp32'
        update_tensor = tf.cast(opt_gather, dtype=tf.float32, name=cast_name)
        if opt_dtype == 'INT8':
            rescale_name = f'{ptm["gather"]}/rescale'
            scale_var = update_dict[embed_node.name][1]
            update_tensor = tf.multiply(update_tensor, scale_var, name=rescale_name)
        update_op_inputs(session.graph, {ptm['gather']: update_tensor})

    return update_dict


def optimize(model_path, save_path, opt_config=None, data_type='BF16', calib_file=None):
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

        def _extract_sub_graph(outputs):
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            util.remove_underscore_class_attr(graph_def)
            return tf.graph_util.extract_sub_graph(graph_def, outputs)

        def _get_all_variables():
            all_variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            all_variables += ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS)
            return all_variables

        # Create Saver
        sub_graph_def = _extract_sub_graph(model_outputs)
        node_names = [node.name for node in sub_graph_def.node]
        save_variables = [v for v in _get_all_variables() if _nd(v.name) in node_names]
        init_name = tf.variables_initializer(save_variables).name
        saver = tf_saver.Saver(save_variables, sharded=True, allow_empty=True)
        tmp_path = tempfile.mkdtemp(dir='.')
        tmp_variable_path = f'{tmp_path}/variables'
        saver.save(sess, tmp_variable_path, write_meta_graph=False, write_state=False)
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
        restore_feed_dict = {new_mgd.saver_def.filename_tensor_name: tmp_variable_path}
        sess.run(new_mgd.saver_def.restore_op_name, restore_feed_dict)
        shutil.rmtree(tmp_path)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # Update assets file
        assets_collection = None
        asset_dict = loader_impl.get_asset_tensors(model_path, meta_graph_def)
        if asset_dict is not None:
            for tensor_name, filename in asset_dict.items():
                asset_op = sess.graph.get_operation_by_name(_nd(tensor_name))
                ts_proto = TensorProto(
                    tensor_shape=asset_op.get_attr('value').tensor_shape,
                    dtype=asset_op.get_attr('value').dtype,
                    string_val=[filename],
                )
                asset_op._set_attr('value', tf.AttrValue(tensor=ts_proto))
                tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_op.outputs[0])
            assets_collection = tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)

        main_op = sess.graph.get_operation_by_name(init_op.name) if init_op else None
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=new_mgd.meta_info_def.tags,
            signature_def_map=new_mgd.signature_def,
            assets_collection=assets_collection,
            main_op=main_op,
        )
        builder.save()

        print('Optmization Result:')
        for key, value in embed_opt_dict.items():
            print(f'Optimize embedding to {value[-1]}: {key}')
        for key, value in dense_opt_dict.items():
            print(f'Optimize dense op to {value[-1]}: {key}')


if __name__ == '__main__':
    model_path = sys.argv[1]
    save_path = sys.argv[2]
    data_type = sys.argv[3]
    calib_file = sys.argv[4] if len(sys.argv) > 4 else None
    opt_dict = None
    optimize(model_path, save_path, opt_dict, data_type, calib_file)
