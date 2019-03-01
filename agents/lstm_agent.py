import tensorflow as tf

class LSTM_Agent(object):
    def __init__(self, is_training, learning_rate, num_layers, n_steps, input_size, output_size, cell_size, batch_size,
                 keep_prob, dropout_in):
        self.is_training = is_training
        self.n_steps = n_steps
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.dropout_in = dropout_in

        with tf.variable_scope('input'):
            self.add_input_layer()
        with tf.variable_scope('hidden'):
            self.add_LSTM_layer()
        with tf.variable_scope('out'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_loss()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def add_input_layer(self):
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [self.n_steps, self.input_size], name='x')
            self.y = tf.placeholder(tf.float32, [self.n_steps, self.output_size], name='y')

        # reshape input to (batch*n_step, in_size)
        in_to_hidden = tf.reshape(self.x, [-1, self.input_size], name='reshape_batch_in')
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        bs_in = self._bias_variable([self.cell_size])
        with tf.name_scope('matmul_in'):
            in_to_hidden = tf.matmul(in_to_hidden, Ws_in) + bs_in

        # reshape input to (batch, n_steps, cell_size)
        self.in_to_hidden = tf.reshape(in_to_hidden, [-1, self.n_steps, self.cell_size], name='unshape_batch_in')
        self.in_to_hidden = tf.layers.dropout(self.in_to_hidden, rate=self.dropout_in, name='in_to_hidden')

    def add_LSTM_layer(self):
        cell = tf.contrib.rnn.LSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(cell, self.in_to_hidden, dtype=tf.float32, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        hidden_to_out = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='reshape_batch_out')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size])

        # shape = (batch * steps, output_size)
        with tf.name_scope('matmul_out'):
            self.pred = tf.matmul(hidden_to_out, Ws_out) + bs_out

    def compute_loss(self):
        y = tf.reshape(self.y, [-1, self.output_size])
        self.loss = tf.reduce_mean(self.ms_error(self.pred, y), name='losses')
        with tf.name_scope('avg_cost'):
            tf.summary.scalar('cost', self.loss)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
