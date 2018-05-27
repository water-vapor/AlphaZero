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


def batch_norm(x, config, is_train=True, scope="bn", data_format="NHWC"):
    with tf.variable_scope(scope):
        return tf.contrib.layers.batch_norm(
            x, decay=config["batch_decay"], center=True, scale=False, is_training=is_train, fused=True, data_format=data_format)


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
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, axis=0)
            grads.append(expanded_g)

        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
