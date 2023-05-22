"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from sparse_operation_kit import experiment as sok


if __name__ == "__main__":

    hvd.init()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess =  tf.compat.v1.Session(config=config)
    sok.init()

    rows = [65536 * 10, 65536]
    cols = [128, 4]
    hotness = [10, 3]
    combiners = ["sum", "sum"]
    batch_size = 65536
    iters = 100
    gpus = [0, min(1, hvd.size() - 1)]

    # initial value of embedding table
    weights = []
    for i in range(len(rows)):
        weight = tf.placeholder(shape=(rows[i],cols[i]), dtype=tf.float32)
        # make sure the weight is same on each rank
        weight = hvd.allreduce(weight)
        weights.append(weight)

    weights_numpy = []
    for i in range(len(rows)):
        weight_numpy = np.ones((rows[i], cols[i]), dtype=np.float32)
        weights_numpy.append(weight_numpy)

    # sok variables
    sok_vars = []
    for i, w in enumerate(weights):
        v = tf.get_embedding_variable("var_{}".format(i),
                                    embedding_dim=cols[i],
                                    initializer=tf.constant_initializer(value=1))
        v.target_gpu = -1
        sok_vars.append(v)

    tf_vars = [tf.Variable(w, use_resource=True) for w in weights]
    '''
    for i, row in enumerate(rows):
       if hvd.rank() == gpus[i]:
           indices = np.arange(row)
           local_indices.append(indices)
       else:
           local_indices.append(None)
   '''

    # indices
    offsets_numpy = []
    values_numpy = []
    offsets = []
    values = []
    indices = []
    for i in range(len(rows)):
        offset_np = np.random.randint(1, hotness[i] + 1, iters * batch_size)
        offsets_numpy.append(offset_np)
        values_np = np.random.randint(0, rows[i], np.squeeze(np.sum(offset_np)))
        values_numpy.append(values_np)
        offset = tf.placeholder(shape=[None], dtype=tf.int64)
        offsets.append(offset)
        value = tf.placeholder(shape=[None], dtype=tf.int64)
        values.append(value)
        indice = tf.RaggedTensor.from_row_lengths(value,offset)
        indices.append(indice)


    #local_indices = []
    left = batch_size // hvd.size() * hvd.rank()
    right = batch_size // hvd.size() * (hvd.rank() + 1)

    # initialize optimizer
    optimizer = tf.train.AdagradOptimizer(0.1)

    def step(params):
        embeddings = sok.lookup_sparse(params, indices, combiners=combiners)
        loss = 0
        for i in range(len(embeddings)):
            loss = loss + tf.reduce_sum(embeddings[i])
        grads = tf.gradients(loss, params)
        optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss, embeddings, grads

    # Do training
    loss1 = []
    ts = []
    t = time.time()

    init_op =  tf.compat.v1.global_variables_initializer()
    sess.run(init_op,feed_dict = {weights[0]:weights_numpy[0],weights[1]:weights_numpy[1]})

    for i in range(iters):
        ts.append(time.time() - t)
        t = time.time()
        tmp_offset_numpy = []
        tmp_values_numpy = []
        for j in range(len(rows)):
        #FIX: tf.RaggedTensor in tensorflow1 can't split by index  , so all card do same data
            tmp_offset_numpy.append(offsets_numpy[j][i * batch_size + left : i * batch_size + right])
            tmp_value_left_offset = np.squeeze(np.sum(offsets_numpy[j][0:i * batch_size + left]))
            tmp_value_rigth_offset = np.squeeze(np.sum(offsets_numpy[j][0:i * batch_size + right]))
            tmp_values_numpy.append(values_numpy[j][tmp_value_left_offset : tmp_value_rigth_offset])
        loss, embeddings, grads = sess.run(step(sok_vars),feed_dict = {offsets[0]:tmp_offset_numpy[0],offsets[1]:tmp_offset_numpy[1],
                                                             values[0]:tmp_values_numpy[0],values[1]:tmp_values_numpy[1]})
        loss1.append(loss)
        print("-" * 30 + "iteration %d" % i + "-" * 30)
        print("loss:", loss)
        print('embeddings', embeddings)
        print('grads', grads)
    out1 = []
    for i in range(len(sok_vars)):
        out1.append(sok_vars[i].eval(sess))
    
    loss2 = []
    def step2(params):
        embeddings = []
        loss = 0
        for i in range(len(params)):
            embedding = tf.nn.embedding_lookup_sparse(
                params[i], indices[i].to_sparse(), None, combiner=combiners[i]
            )
            embeddings.append(embedding)
            loss = loss + tf.reduce_sum(embedding)
        grads = tf.gradients(loss, params)
        grads = [hvd.allreduce(grad, op=hvd.Sum) for grad in grads]
        optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss, embeddings, grads

    for i in range(iters):
        tmp_offset_numpy = []
        tmp_values_numpy = []
        for j in range(len(rows)):
            #FIX: tf.RaggedTensor in tensorflow1 can't split by index  , so all card do same data
            tmp_offset_numpy.append(offsets_numpy[j][i * batch_size + left : i * batch_size + right])
            tmp_value_left_offset = np.squeeze(np.sum(offsets_numpy[j][0:i * batch_size + left]))
            tmp_value_rigth_offset = np.squeeze(np.sum(offsets_numpy[j][0:i * batch_size + right]))
            tmp_values_numpy.append(values_numpy[j][tmp_value_left_offset : tmp_value_rigth_offset])
        loss, embeddings, grads = sess.run(step2(tf_vars),feed_dict = {offsets[0]:tmp_offset_numpy[0],offsets[1]:tmp_offset_numpy[1],
                                                             values[0]:tmp_values_numpy[0],values[1]:tmp_values_numpy[1]})
        loss2.append(loss)
        print("-" * 30 + "iteration %d" % i + "-" * 30)
        print("tf loss:", loss)
        print("tf embeddings:", embeddings)
        print("tf grads:", grads)
    out2 = []
    for i, v in enumerate(tf_vars):
        if hvd.rank() == gpus[i]:
            out2.append(v.eval(sess))
        else:
            out2.append(None)

    # Check results
    diff = 0
    for i in range(iters):
        # normalize
        length = loss1[i] ** 2 + loss2[i] ** 2 + 1e-8
        print('loss ', loss1[i], loss2[i])
        diff = diff + (loss1[i] - loss2[i]) ** 2 / length
    print("[SOK INFO] loss diff:", diff)
    assert diff < 1e-8

    print("[SOK INFO] lookup_sparse distributed test passed")
    ts = ts[5:]
    print("[SOK INFO] Average time: %f ms/iteration" % (sum(ts) / len(ts) * 1000))
    sess.close()
