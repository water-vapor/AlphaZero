import tensorflow as tf


def batch_split(num, *args):
    ress = []
    for arg in args:
        res = []
        batch = arg.shape[0]
        piece = batch // num
        for idx in range(num):
            start = idx * piece
            end = (idx + 1) * piece if idx != num - 1 else batch
            res.append(arg[start: end])
        ress.append(res)
    return zip(*ress)


def get_variable_on_cpu(name, shape, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None):
    with tf.device("/CPU:0"):
        return tf.get_variable(name, shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=trainable, collections=collections)


def batch_norm(x, config, is_train=True, scope="bn", mode="NHWC"):
    with tf.variable_scope(scope):
        res = tf.contrib.layers.batch_norm(
            x, decay=config["batch_decay"], center=False, scale=False, is_training=is_train, fused=True, data_format=mode)
        dim = x.get_shape()[-1]
        w = get_variable_on_cpu(
            "w", [dim], initializer=tf.constant_initializer(1.))
        b = get_variable_on_cpu(
            "b", [dim], initializer=tf.constant_initializer(0.))
        return res * w + b


def linear(x, dim, bias, bias_start=0., scope="linear"):
    with tf.variable_scope(scope):
        input_dim = x.get_shape().as_list()[-1]
        W = get_variable_on_cpu("W", [input_dim, dim])
        res = tf.matmul(x, W)
        if not bias:
            return res
        b = get_variable_on_cpu(
            "b", [dim], initializer=tf.constant_initializer(bias_start))
        return tf.nn.bias_add(res, b)


def average_gradients(tower_grads, num_gpu=1):
    average_grads = []
    gpu_idx = 0
    for grad_and_vars in zip(*tower_grads):

        grads = [grad_and_var[0] for grad_and_var in grad_and_vars]

        with tf.device("/GPU:{}".format(gpu_idx)):
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        gpu_idx = (gpu_idx + 1) % num_gpu
    return average_grads
