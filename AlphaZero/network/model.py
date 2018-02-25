import tensorflow as tf

from AlphaZero.network.util import batch_norm, linear


def get_multi_models(num_gpu, config, game_config, mode="NHWC"):
    models = []
    with tf.variable_scope("models"):
        for idx in range(num_gpu):
            with tf.name_scope("model_{}".format(idx)) as scope, tf.device("/GPU:{}".format(idx)):
                model = Model(config, scope, game_config, mode=mode)
                tf.get_variable_scope().reuse_variables()
                models.append(model)
    return models


class Model(object):
    def __init__(self, config, scope, game_config, mode="NHWC"):
        self.config = config
        self.scope = scope
        self.mode = mode
        self._game_config = game_config
        self._w = game_config['board_width']
        self._h = game_config['board_height']
        self._f = game_config['history_step'] * \
            game_config['planes_per_step'] + game_config['additional_planes']

        self.x = tf.placeholder(
            tf.float32, [None, self._f, self._w, self._h], name="x")
        if game_config['output_plane'] == 1:
            self.p = tf.placeholder(
                tf.float32, [None, game_config['flat_move_output']], name="p")
        else:
            # TODO: multiple layer output needs further modification on model structure
            self.p = tf.placeholder(tf.float32, [None, game_config['output_plane'], game_config['flat_move_output']],
                                    name="p")
        self.v = tf.placeholder(tf.float32, [None], name="v")
        self.is_train = tf.placeholder(tf.bool, [], name="is_train")

        self.global_step = tf.get_variable(
            "global_step", [], tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        self._build_forward()
        self._build_loss()

    def _build_forward(self):
        config = self.config
        _activation = tf.nn.relu
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=float(self.config["l2"]))

        if self.mode == "NHWC":
            inputs = tf.transpose(self.x, [0, 2, 3, 1])
        else:
            inputs = self.x

        W0 = tf.get_variable(
            "W0", [3, 3, self._f, 256], regularizer=regularizer)
        R = tf.nn.conv2d(inputs, W0, strides=[
            1, 1, 1, 1], padding='SAME', data_format=self.mode)
        R = _activation(batch_norm(R, config, self.is_train, mode=self.mode))

        for layer in range(config["num_blocks"]):
            with tf.variable_scope("resblock_{}".format(layer)):
                W1 = tf.get_variable(
                    "W1", [3, 3, 256, 256], regularizer=regularizer)
                W2 = tf.get_variable(
                    "W2", [3, 3, 256, 256], regularizer=regularizer)
                R1 = tf.nn.conv2d(
                    R, W1, strides=[1, 1, 1, 1], padding='SAME', data_format=self.mode)
                R1 = _activation(batch_norm(
                    R1, config, self.is_train, scope="B1", mode=self.mode))
                R2 = tf.nn.conv2d(
                    R1, W2, strides=[1, 1, 1, 1], padding='SAME', data_format=self.mode)
                R2 = batch_norm(R2, config, self.is_train, scope="B2")
                R = _activation(tf.add(R, R2))

        with tf.variable_scope("policy_head"):
            W0 = tf.get_variable("W0", [1, 1, 256, 2], regularizer=regularizer)
            R_p = tf.nn.conv2d(
                R, W0, strides=[1, 1, 1, 1], padding='SAME', data_format=self.mode)
            R_p = tf.reshape(_activation(batch_norm(
                R_p, config, self.is_train, mode=self.mode)), [-1, self._w * self._h * 2])
            logits = linear(R_p, self._game_config['flat_move_output'], True)
            R_p = tf.nn.softmax(logits)

        with tf.variable_scope("value_head"):
            W0 = tf.get_variable("W0", [1, 1, 256, 1], regularizer=regularizer)
            R_v = tf.nn.conv2d(
                R, W0, strides=[1, 1, 1, 1], padding='SAME', data_format=self.mode)
            R_v = tf.reshape(_activation(batch_norm(
                R_v, config, self.is_train, mode=self.mode)), [-1, self._w * self._h])
            R_v = _activation(linear(R_v, 256, True, scope="F1"))
            R_v = tf.nn.tanh(tf.squeeze(
                linear(R_v, 1, True, scope="F2"), [-1]))

        self.logits = logits
        self.R_p = R_p
        self.R_v = R_v

    def _build_loss(self):
        v_loss = tf.reduce_mean(tf.square(self.R_v - self.v))
        p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.p))
        r_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = p_loss + v_loss * self.config["MSE_scaling"] + r_loss

    def get_loss(self):
        return self.loss
