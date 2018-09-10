import json
import numpy as np

import tensorflow as tf
import utils.distributions as distributions

from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    ops.reset_default_graph()
    print("graph successfully reset")

class BasicLSTM(object):
    def __init__(self, batch_size=50,
                 sequence_length=8,
                 num_units=128,
                 embedding_size=128,
                 learning_rate=0.003,
                 grad_clip=10,
                 mode='train',
                 reuse=False):

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_units = num_units
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.mode = mode
        # ensure all ops are from the same graph
        with tf.variable_scope("basic_lstm", reuse=reuse):
            self.g = tf.Graph()
            with self.g.as_default():
                self._build_graph()

        self._init_session()

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        '''Method that builds the graph as per our blog post.'''

        cell = rnn_cell.BasicLSTMCell(self.num_units, state_is_tuple=True)

        self.input_data = tf.placeholder(tf.float32, [None, self.sequence_length, 2])
        self.target_data = tf.placeholder(tf.float32, [None, self.sequence_length, 2])

        self.lr = tf.Variable(self.learning_rate, trainable=False, name="learning_rate")
        self.initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        
        # input dimensionality is the x and y position at every step
        # the output is comprised of two means, two std and 1 corr variable
        embedding_w, embedding_b, output_w, output_b = self.build_embeddings(input_dim=2, output_dim=5)

        # Prepare inputs ..
        inputs = tf.split(self.input_data, self.sequence_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # the actual LSTM model
        embedded_inputs = self.embed_inputs(inputs, embedding_w, embedding_b)
        outputs, last_state = self.lstm_advance(embedded_inputs, cell)
        final_output = self.final_layer(outputs, output_w, output_b)
        
        self.final_state = last_state
        # reshape target data so that it aligns with predictions
        flat_target_data = tf.reshape(self.target_data, [-1, 2])
        # Extract the x-coordinates and y-coordinates from the target data
        [x_data, y_data] = tf.split(flat_target_data, 2, 1)

        # Extract coef from output of the linear output layer
        [o_mux, o_muy, o_sx, o_sy, o_corr] = self.get_coef(final_output)
        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.name_scope("Optimization"):
                lossfunc = self.get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

                self.cost = tf.div(lossfunc, (self.batch_size * self.sequence_length))
                trainable_params = tf.trainable_variables()

                # apply L2 regularisation
                l2 = 0.05 * sum(tf.nn.l2_loss(t_param) for t_param in trainable_params)
                self.cost = self.cost + l2
                tf.summary.scalar('cost', self.cost)

                self.gradients = tf.gradients(self.cost, trainable_params)
                grads, _ = tf.clip_by_global_norm(self.gradients, self.grad_clip)

                # Adam might also do a good job as in Graves (2013)
                optimizer = tf.train.RMSPropOptimizer(self.lr)
                # Train operator
                self.train_op = optimizer.apply_gradients(zip(grads, trainable_params))

                self.init = tf.global_variables_initializer()

    def build_embeddings(self, input_dim, output_dim):
        # Define variables for embedding the input
        with tf.variable_scope("coordinate_embedding"):
            embedding_w = tf.get_variable("embedding_w", [input_dim, self.embedding_size])
            embedding_b = tf.get_variable("embedding_b", [self.embedding_size])

        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            output_w = tf.get_variable("output_w", [self.num_units, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=True)
            output_b = tf.get_variable("output_b", [output_dim], initializer=tf.constant_initializer(0.1), trainable=True)

        # # # Used for tensorboard summary # # #
        tf.summary.histogram('embedding_w', embedding_w)
        tf.summary.histogram('embedding_b', embedding_b)
        tf.summary.histogram('output_w', output_w)
        tf.summary.histogram('output_b', output_b)
        # # # # # # # # # # # # # # # # # # # #

        return embedding_w, embedding_b, output_w, output_b

    def embed_inputs(self, inputs, embedding_w, embedding_b):
        # embed the inputs
        with tf.name_scope("Embed_inputs"):
            embedded_inputs = []
            for x in inputs:
                # Each x is a 2D tensor of size numPoints x 2
                embedded_x = tf.nn.relu(tf.add(tf.matmul(x, embedding_w), embedding_b))
                embedded_inputs.append(embedded_x)

            return embedded_inputs

    def final_layer(self, outputs, output_w, output_b):
        with tf.name_scope("Final_layer"):
            # Apply the linear layer. Output would be a 
            # tensor of shape 1 x output_size
            output = tf.reshape(tf.concat(outputs, 1), [-1, self.num_units])
            output = tf.nn.xw_plus_b(output, output_w, output_b)
            return output

    def lstm_advance(self, embedded_inputs, cell, scope_name="LSTM"):
        # advance the lstm cell state with one for each entry
        with tf.variable_scope(scope_name) as scope:
            state = self.initial_state
            outputs = []
            for i, inp in enumerate(embedded_inputs):
                if i > 0:
                    scope.reuse_variables()
                output, last_state = cell(inp, state)
                outputs.append(output)

            return outputs, last_state

    def get_lossfunc(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        # Calculate the PDF of the data w.r.t to the distribution
        result0 = distributions.tf_2d_normal(self.g, x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
        # For numerical stability purposes as in Vemula (2018)
        epsilon = 1e-20
        # Numerical stability
        result1 = -tf.log(tf.maximum(result0, epsilon))

        return tf.reduce_sum(result1)

    def get_coef(self, output):
        # eq 20 -> 22 of Graves (2013)
        z = output
        z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, 1)

        # The output must be exponentiated for the std devs
        z_sx = tf.exp(z_sx)
        z_sy = tf.exp(z_sy)
        # Tanh applied to keep it in the range [-1, 1]
        z_corr = tf.tanh(z_corr)

        return [z_mux, z_muy, z_sx, z_sy, z_corr]

    # Thanks D. Ha!
    # modified from https://github.com/hardmaru/WorldModelsExperiments
    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p*10000).astype(np.int).tolist() # ..?!
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                pshape = self.sess.run(var).shape
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op = var.assign(p.astype(np.float)/10000.)
                self.sess.run(assign_op)
                idx += 1
        
    def save_json(self, jsonfile='lstm.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def load_json(self, jsonfile='lstm.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
