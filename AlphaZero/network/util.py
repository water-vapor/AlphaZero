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


def batch_norm(x, config, is_train=True, scope="bn", mode="NHWC"):
    with tf.variable_scope(scope):
        res = tf.contrib.layers.batch_norm(
            x, decay=config["batch_decay"], center=False, scale=False, is_training=is_train, fused=True, data_format=mode)
        dim = x.get_shape()[-1]
        w = tf.get_variable(
            "w", [dim], initializer=tf.constant_initializer(1.))
        b = tf.get_variable(
            "b", [dim], initializer=tf.constant_initializer(0.))
        return res * w + b


def linear(x, dim, bias, bias_start=0., scope="linear"):
    with tf.variable_scope(scope):
        input_dim = x.get_shape().as_list()[-1]
        W = tf.get_variable("W", [input_dim, dim])
        res = tf.matmul(x, W)
        if not bias:
            return res
        b = tf.get_variable(
            "b", [dim], initializer=tf.constant_initializer(bias_start))
        return tf.nn.bias_add(res, b)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        grads = [grad_and_var[0] for grad_and_var in grad_and_vars]

        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
