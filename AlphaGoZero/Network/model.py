import tensorflow as tf
from Network.util import batch_norm, linear


def get_multi_models(num_gpu, config):
    models = []
    with tf.variable_scope("models"):
        for idx in range(num_gpu):
            with tf.name_scope("model_{}".format(idx)) as scope, tf.device("/GPU:{}".format(idx)):
                model = Model(config, scope)
                tf.get_variable_scope().reuse_variables()
                models.append(model)
    return models


class Model(object):
    def __init__(self, config, scope):
        self.config = config
        self.scope = scope

        self.x = tf.placeholder(tf.float32, [None, 17, 19, 19], name="x")
        self.p = tf.placeholder(tf.float32, [None, 362], name="p")
        self.v = tf.placeholder(tf.float32, [None], name="v")
        self.is_train = tf.placeholder(tf.bool, [], name="is_train")

        self.global_step = tf.get_variable(
            "global_step", [], tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        self._build_forward()
        self._build_loss()

    def _build_forward(self):
        config = self.config
        _activation = tf.nn.relu
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.l2)

        inputs = tf.transpose(self.x, [0, 2, 3, 1])
        W0 = tf.get_variable(
            "W0", [3, 3, 19 * 19 * 17, 256], regularizer=regularizer)
        R = tf.nn.conv2d(inputs, W0, strides=[1, 1, 1, 1], padding='SAME')
        R = _activation(batch_norm(R, config, self.is_train))

        for i in range(config.num_blocks):
            with tf.variable_scope("resblock_{}".format(i)):
                W1 = tf.get_variable(
                    "W1", [3, 3, 256, 256], regularizer=regularizer)
                W2 = tf.get_variable(
                    "W2", [3, 3, 256, 256], regularizer=regularizer)
                R1 = tf.nn.conv2d(R, W1, strides=[1, 1, 1, 1], padding='SAME')
                R1 = _activation(batch_norm(
                    R1, config, self.is_train, scope="B1"))
                R2 = tf.nn.conv2d(R1, W2, strides=[1, 1, 1, 1], padding='SAME')
                R2 = batch_norm(R2, config, self.is_train, scope="B2")
                R = _activation(tf.add(R, R2))

        with tf.variable_scope("policy_head"):
            W0 = tf.get_variable("W0", [1, 1, 256, 2], regularizer=regularizer)
            R_p = tf.nn.conv2d(R, W0, strides=[1, 1, 1, 1], padding='SAME')
            R_p = tf.reshape(_activation(batch_norm(
                R_p, config, self.is_train)), [-1, 19 * 19 * 2])
            R_p = tf.nn.softmax(linear(R_p, 362, True))

        with tf.variable_scope("value_head"):
            W0 = tf.get_variable("W0", [1, 1, 256, 1], regularizer=regularizer)
            R_v = tf.nn.conv2d(R, W0, strides=[1, 1, 1, 1], padding='SAME')
            R_v = tf.reshape(_activation(batch_norm(
                R_v, config, self.is_train)), [-1, 19 * 19])
            R_v = _activation(linear(R_v, 256, True, scope="F1"))
            R_v = tf.nn.tanh(tf.squeeze(
                linear(R_v, 1, True, scope="F2"), [-1]))

        self.R_p = R_p
        self.R_v = R_v

    def _build_loss(self):
        v_loss = tf.reduce_mean(tf.square(self.R_v - self.v))
        p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.R_p, labels=self.p))
        r_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = p_loss + v_loss + r_loss

    def get_loss(self):
        return self.loss
