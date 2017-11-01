import tensorflow as tf
import numpy as np

from Network.model import get_multi_models
from Network.util import average_gradients

config = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("num_blocks", 19, "number of residual blocks in the model")
tf.app.flags.DEFINE_float("batch_decay", 0.9, "decay factor of batch normalization")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "lr of momentum optimizer")
tf.app.flags.DEFINE_float("momentum", 0.9, "momentum of momentum optimizer")
tf.app.flags.DEFINE_float("l2", 1e-4, "L2 regularization factor")


class Network(object):

    def __init__(self, num_gpu=1):
        self.num_gpu = num_gpu
        self.models = get_multi_models(self.num_gpu, config)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.opt = tf.train.MomentumOptimizer(
            config.learning_rate, momentum=config.momentum)

        loss_list = []
        grad_list = []
        p_list = []
        v_list = []
        for idx, model in enumerate(self.models):
            with tf.name_scope("grad_{}".format(idx)), tf.device("/GPU:{}".format(idx)):
                loss = model.get_loss()
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grad = self.opt.compute_gradients(loss)
                loss_list.append(loss)
                grad_list.append(grad)
                p_list.append(model.R_p)
                v_list.append(model.R_v)

        self.loss = tf.add_n(loss_list) / len(loss_list)
        self.grad = average_gradients(grad_list)
        self.train_op = self.opt.apply_gradients(
            self.grad, global_step=self.models[0].global_step)
        self.R_p = tf.concat(p_list, axis=0)
        self.R_v = tf.concat(v_list, axis=0)

    def update(self, data):
        '''
        Update the model parameters.

        Input: (state, action, result)
               state: numpy array of shape [None, 17, 19, 19]
               action: numpy array of shape [None, 362]
               result: numpy array of shape [None]
        Return: Average loss of the batch

        '''
        feed_dict = {}
        batch = len(data[0].shape[0])
        piece = batch / self.num_gpu
        for idx, model in enumerate(self.models):
            start_idx = idx * piece
            end_idx = (idx + 1) * piece
            if idx == self.num_gpu - 1:
                end_idx = batch
            feed_dict[model.x] = data[0][start_idx: end_idx]
            feed_dict[model.p] = data[1][start_idx: end_idx]
            feed_dict[model.v] = data[2][start_idx: end_idx]
            feed_dict[model.is_train] = True
        loss, train_op = self.sess.run(
            [self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def response(self, data):
        '''
        Predict the action and result given current state.

        Input: (state, )
               state: numpy array of shape [None, 17, 19, 19]
        Return: (R_p, R_v)
                R_p: probability distribution of action, numpy array of shape [None, 362]
                R_v: expected value of current state, numpy array of shape [None]
        '''
        feed_dict = {}
        batch = len(data[0].shape[0])
        piece = batch / self.num_gpu
        for idx, model in enumerate(self.models):
            start_idx = idx * piece
            end_idx = (idx + 1) * piece
            if idx == self.num_gpu - 1:
                end_idx = batch
            feed_dict[model.x] = data[0][start_idx: end_idx]
            feed_dict[model.is_train] = False
        R_p, R_v = self.sess.run([self.R_p, self.R_v], feed_dict=feed_dict)
        return R_p, R_v

    def get_global_step(self):
        return self.sess.run(self.models[0].global_step) + 1

    def save(self, filename):
        global_step = self.get_global_step()
        self.saver.save(self.sess, filename, global_step=global_step)

    def load(self, filename):
        self.saver.restore(self.sess, filename)
