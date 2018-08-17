import tensorflow as tf
import numpy as np


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator:
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout  input_x : batch_size * seq_len
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # input is word idx
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # batch * num_class (1 / 0)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        with tf.variable_scope("discriminator"):
            # Embedding:
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # batch_size * seq * embedding_size
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            pooled_outputs = list()
            # Create a convolution + maxpool layer for each filter size
            for filter_size, filter_num in zip(filter_sizes, num_filters):
                with tf.name_scope("cov2d-maxpool%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, filter_num]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # print(conv.name, ": ", conv.shape) batch * (seq - filter_shape) + 1 * 1(output channel) *
                    # filter_num
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")  # 全部池化到 1x1
                    # print(conv.name, ": ", conv.shape , "----", pooled.name, " : " ,pooled.shape)
                    pooled_outputs.append(pooled)
            total_filters_num = sum(num_filters)

            self.h_pool = tf.concat(pooled_outputs, 3)
            # print(self.h_pool.shape) # batch * 1 * 1 * total_filters_num
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters_num])  # batch * total_num

            # remove highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # add droppout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([total_filters_num, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
                self.loss = losses + l2_reg_lambda * l2_loss
            with tf.name_scope("accuracy"):
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(self.predictions, tf.argmax(self.input_y, 1)), tf.float32))
        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        # aggregation_method =2 能够帮助减少内存占用
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    @staticmethod
    def restore(sess, saver, path):
        saver.restore(sess, save_path=path)
