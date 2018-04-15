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


def batch_norm(x, config, is_train=True, scope="bn", mode="NHWC", reuse=False):
    with tf.variable_scope(scope) as variable_scope:
        return tf.contrib.layers.batch_norm(x, decay=config["batch_decay"], center=True, scale=True, is_training=is_train, fused=True, data_format=mode, reuse=reuse, scope=variable_scope)


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
        grads = []
        for g, var in grad_and_vars:
            assert g is not None, var.name
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
