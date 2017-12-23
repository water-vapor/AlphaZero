import os
import numpy as np
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf

from AlphaGoZero.Network.model import get_multi_models
from AlphaGoZero.Network.util import average_gradients

reinforce_config = os.path.join("AlphaGoZero", "Network", "reinforce.yaml")


class Network(object):

    def __init__(self, num_gpu=1, config_file=reinforce_config, pretrained=False, mode="NHWC", cluster=None, job=None):
        with open(config_file) as fh:
            self.config = yaml.load(fh)
        self.num_gpu = num_gpu

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        server = tf.train.Server(cluster, job_name=job, task_index=0, config=sess_config)
        self.sess = tf.Session(target=server.target)

        with tf.device('/job:'+job+'/task:0'):
            self.models = get_multi_models(self.num_gpu, self.config, mode=mode)
            self.saver = tf.train.Saver()

            self.lr = tf.placeholder(tf.float32, [], name="lr")
            self.opt = tf.train.MomentumOptimizer(
                self.lr, momentum=self.config["momentum"])

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

        if pretrained:
            self.saver.restore(
                self.sess, tf.train.latest_checkpoint(self.config.save_dir))
        else:
            self.sess.run(tf.global_variables_initializer())

    def update(self, data):
        '''
        Update the model parameters.

        Input: (state, action, result)
               state: numpy array of shape [None, 17, 19, 19]
               action: numpy array of shape [None, 362]
               result: numpy array of shape [None]
        Return: Average loss of the batch

        '''
        global_step = self.get_global_step()
        if global_step % 1000 == 0:
            learning_scheme = self.config["learning_rate"]
            divides = sorted(list(learning_scheme))
            current = 0
            for element in divides:
                if element < global_step:
                    current = element
            self.learning_rate = learning_scheme[current]

        feed_dict = {}
        batch = data[0].shape[0]
        piece = batch // self.num_gpu
        for idx, model in enumerate(self.models):
            start_idx = idx * piece
            end_idx = (idx + 1) * piece
            if idx == self.num_gpu - 1:
                end_idx = batch
            feed_dict[model.x] = data[0][start_idx: end_idx]
            feed_dict[model.p] = data[1][start_idx: end_idx]
            feed_dict[model.v] = data[2][start_idx: end_idx]
            feed_dict[model.is_train] = True
        feed_dict[self.lr] = self.learning_rate
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
        batch = data[0].shape[0]
        piece = batch // self.num_gpu
        for idx, model in enumerate(self.models):
            start_idx = idx * piece
            end_idx = (idx + 1) * piece
            if idx == self.num_gpu - 1:
                end_idx = batch
            feed_dict[model.x] = data[0][start_idx: end_idx]
            feed_dict[model.is_train] = False
        R_p, R_v = self.sess.run([self.R_p, self.R_v], feed_dict=feed_dict)
        return R_p, R_v

    def evaluate(self, data):
        '''
        Calculate loss and result based on supervised data.

        Input: (state, action, result)
               state: numpy array of shape [None, 17, 19, 19]
               action: numpy array of shape [None, 362]
               result: numpy array of shape [None]
        Return: (loss, R_p, R_v)
                loss: average loss of the batch
                R_p: probability distribution of action, numpy array of shape [None, 362]
                R_v: expected value of current state, numpy array of shape [None]
        '''
        feed_dict = {}
        batch = data[0].shape[0]
        piece = batch // self.num_gpu
        for idx, model in enumerate(self.models):
            start_idx = idx * piece
            end_idx = (idx + 1) * piece
            if idx == self.num_gpu - 1:
                end_idx = batch
            feed_dict[model.x] = data[0][start_idx: end_idx]
            feed_dict[model.p] = data[1][start_idx: end_idx]
            feed_dict[model.v] = data[2][start_idx: end_idx]
            feed_dict[model.is_train] = False
        loss, R_p, R_v = self.sess.run(
            [self.loss, self.R_p, self.R_v], feed_dict=feed_dict)
        return loss, R_p, R_v

    def get_global_step(self):
        return self.sess.run(self.models[0].global_step)

    def save(self, filename):
        global_step = self.get_global_step()
        self.saver.save(self.sess, filename, global_step=global_step)

    def load(self, filename):
        self.saver.restore(self.sess, filename)
