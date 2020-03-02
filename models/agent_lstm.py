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

class AgentLSTM(object):
    def __init__(self, batch_size=50,
                 sequence_length=8,
                 num_units=128,
                 embedding_size=64,
                 learning_rate=0.003,
                 grad_clip=10,
                 max_num_agents=40,
                 input_dim=3,
                 output_dim=3,
                 mode='train',
                 reuse=False):

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units = num_units
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.max_num_agents = max_num_agents
        self.grad_clip = grad_clip
        self.mode = mode
        # ensure all ops are from the same graph
        with tf.variable_scope("agent_lstm", reuse=reuse):
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

        cell = rnn_cell.BasicLSTMCell(self.num_units, state_is_tuple=False)
        self.input_data = tf.placeholder(tf.float32, [self.sequence_length, self.max_num_agents, self.input_dim])
        self.target_data = tf.placeholder(tf.float32, [self.sequence_length, self.max_num_agents, self.output_dim])

        self.lr = tf.Variable(self.learning_rate, trainable=False, name="learning_rate")
        self.output_size = 5
        # at the moment consider a single initial state, then try with one per agent
        # then try the one per intent!
        self.LSTM_states = tf.zeros([self.max_num_agents, cell.state_size], name="LSTM_states") # state size includes both h and z ..
        self.initial_states = tf.split(self.LSTM_states, self.max_num_agents, 0)
        self.output_states = tf.split(tf.zeros([self.max_num_agents, cell.output_size]), self.max_num_agents, 0)

        frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, self.sequence_length, 0)]
        frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, self.sequence_length, 0)]

        nonexistent_agent = tf.constant(0.0, name="zero_agent")
        self.cost = tf.constant(0.0, name="cost")
        self.counter = tf.constant(0.0, name="counter")
        self.increment = tf.constant(1.0, name="increment")
        # input dimensionality is the x and y position at every step
        # the output is comprised of two means, two std and 1 corr variable
        embedding_w, embedding_b, output_w, output_b = self.buildEmbeddings(input_dim=self.input_dim-1, output_dim=5)

        self.initial_output = tf.split(tf.zeros([self.max_num_agents, self.output_size]), self.max_num_agents, 0)
        for seq, frame in enumerate(frame_data):
            print("Frame number", seq)
            current_frame_data = frame  # max_num_agents x 3 tensor
            for agent in range(self.max_num_agents):
                agent_id = current_frame_data[agent, 0]
                self.spatial_input = tf.slice(current_frame_data, [agent, 1], [1, 2])
                embedded_spatial_input = tf.nn.relu(tf.nn.xw_plus_b(self.spatial_input, embedding_w, embedding_b))

                # NOTE: This last state is the last agent's last state ...
                reuse = True if seq > 0 or agent > 0 else False
                self.output_states[agent], self.initial_states[agent] = self.lstmAdvance(embedded_spatial_input, cell, self.initial_states[agent], reuse)
                self.initial_output[agent] = tf.nn.xw_plus_b(self.output_states[agent], output_w, output_b)

                [x_data, y_data] = tf.split(tf.slice(frame_target_data[seq], [agent, 1], [1, 2]), 2, axis=1)
                target_agent_id = frame_target_data[seq][agent, 0]

                [o_mux, o_muy, o_sx, o_sy, o_corr] = self.getCoef(self.initial_output[agent])
                lossfunc = self.getLossFunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
                self.cost = tf.where(
                    tf.logical_or(
                        tf.equal(agent_id, nonexistent_agent),
                        tf.equal(target_agent_id, nonexistent_agent)
                    ), self.cost, tf.add(self.cost, lossfunc))
                self.counter = tf.where(
                    tf.logical_or(
                        tf.equal(agent_id, nonexistent_agent),
                        tf.equal(target_agent_id, nonexistent_agent)
                    ), self.counter, tf.add(self.counter, self.increment))

        self.final_states = tf.concat(self.initial_states, 0)
        self.final_output = self.initial_output
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.name_scope("Optimization"):
                self.cost = tf.divide(self.cost, self.counter)
                tvars = tf.trainable_variables()
                # L2 loss
                l2 = 0.0005*sum(tf.nn.l2_loss(tvar) for tvar in tvars)
                self.cost = self.cost + l2
                self.gradients = tf.gradients(self.cost, tvars)
                grads, _ = tf.clip_by_global_norm(self.gradients, self.grad_clip)
                optimizer = tf.train.RMSPropOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.init = tf.global_variables_initializer()

    def buildEmbeddings(self, input_dim, output_dim):
        # Define variables for embedding the input
        with tf.variable_scope("coordinate_embedding"):
            embedding_w = tf.get_variable("embedding_w", [input_dim, self.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedding_b = tf.get_variable("embedding_b", [self.embedding_size], initializer=tf.constant_initializer(0.1))

        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            output_w = tf.get_variable("output_w", [self.num_units, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
            output_b = tf.get_variable("output_b", [output_dim], initializer=tf.constant_initializer(0.1))

        return embedding_w, embedding_b, output_w, output_b

    def embedInputs(self, inputs, embedding_w, embedding_b):
        # embed the inputs
        with tf.name_scope("Embed_inputs"):
            embedded_inputs = []
            for x in inputs:
                # Each x is a 2D tensor of size numPoints x 2
                embedded_x = tf.nn.relu(tf.add(tf.matmul(x, embedding_w), embedding_b))
                embedded_inputs.append(embedded_x)

            return embedded_inputs

    def finalLayer(self, outputs, output_w, output_b):
        with tf.name_scope("Final_layer"):
            # Apply the linear layer. Output would be a 
            # tensor of shape 1 x output_size
            output = tf.reshape(tf.concat(outputs, 1), [-1, self.num_units])
            output = tf.nn.xw_plus_b(output, output_w, output_b)
            return output

    def lstmAdvance(self, embedded_inputs, cell, state, reuse=False, scope_name="LSTM"):
        # advance the lstm cell state with one for each entry
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            output, last_state = cell(embedded_inputs, state)

            return output, last_state

    def getLossFunc(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        # Calculate the PDF of the data w.r.t to the distribution
        result0 = distributions.tf_2d_normal(self.g, x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
        # For numerical stability purposes as in Vemula (2018)
        epsilon = 1e-20
        # Numerical stability
        result1 = -tf.log(tf.maximum(result0, epsilon))

        return tf.reduce_sum(result1)

    def getCoef(self, output):
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
    def getModelParams(self):
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
        model_params, model_shapes, model_names = self.getModelParams()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def load_json(self, jsonfile='lstm.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
