import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from rhyme import calc_rhyme


class G_beta:
    def __init__(self, lstm, update_rate):
        self.lstm = lstm
        self.update_rate = update_rate

        # copy parameters from lstm model
        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

        # placeholder
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)

        # embedded input
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x),
                                            perm=[1, 0, 2])  # seq * batch * emb_size

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length
        )
        ta_emb_x = ta_emb_x.unstack(self.processed_x)  # seq * emb_size

        ta_x = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=self.sequence_length
        )
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))  # seq * batch

        self.h0 = lstm.h0

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        # while i < given num , using the provided tokens as input
        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x

        def _g_recurrence_2(i, x_t, h_tm1, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)  # logits  : batch x vocab
            log_prob = tf.log(tf.nn.softmax(o_t))  # log prob
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)
            gen_x = gen_x.write(i, next_token)
            return i + 1, x_tp1, h_t, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, self.gen_x)
        )

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

    def get_reward(self, sess, target, input_x, roll_out_num, discriminator):
        rewards = []
        for i in range(roll_out_num):  # sample times
            for given_num in range(1, self.sequence_length):  # 1 -> 19
                feed = {self.x: target, self.given_num: given_num, self.lstm.inputs: input_x}
                samples = sess.run(self.gen_x, feed)
                feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[0] for item in ypred_for_auc])  # probability of being real data
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred
            # the last token reward
            feed = {discriminator.input_x: target, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            # probability of being fake
            ypred = np.array([item[0] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.sequence_length - 1] += ypred  # seq_len * batch_size
        rewards = np.transpose(np.array(rewards)) / (1.0 * roll_out_num)  # batch_size x seq_length
        # baseline = np.mean(rewards, axis=0)
        return rewards

    def get_reward_rhyme(self, sess, input_x, input_y, roll_out_num, discriminator, reward_weight, idx2word):
        """
        calculate sentence reward based on rhyme and discriminator
        Args:
            sess: session to run
            input_x: the x of original x-y pair
            input_y: the generated y_hat corresponding to x, we calculate reward of input_y
            roll_out_num: MC rollout times
            discriminator: discriminator model to evaluate probability of being real
            reward_weight: reward weight of rhyme reward
            idx2word: dict, map word idx to real word, in order to
        """
        rewards = []
        for i in range(roll_out_num):  # sample times
            for given_num in range(1, self.sequence_length):  # 1 -> 19
                feed = {self.x: input_y, self.given_num: given_num}
                # print(input_x.shape)
                samples = sess.run(self.gen_x, feed)
                feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])  # probability of being real data
                # print(ypred.shape)
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred
            # the last token reward
            feed = {discriminator.input_x: input_y, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.sequence_length - 1] += ypred  # seq_len * batch_size
        rewards = np.transpose(np.array(rewards)) / (1.0 * roll_out_num)  # batch_size x seq_length
        rhyme_rewards = calc_rhyme(input_x, input_y, idx2word, reverse=True)
        baseline = np.mean(rewards, axis=0)
        return rewards - baseline, np.mean(rhyme_rewards, axis=0)

    def update_params(self):
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit
        self.g_output_unit = self.update_output_unit

    def create_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor with weight decay
        with tf.variable_scope("lstm_beta"):
            self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.lstm.Wi)
            self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.lstm.Ui)
            self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.lstm.bi)

            self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.lstm.Wf)
            self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.lstm.Uf)
            self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.lstm.bf)

            self.Wog = self.update_rate * self.Wog + (1 - self.update_rate) * tf.identity(self.lstm.Wog)
            self.Uog = self.update_rate * self.Uog + (1 - self.update_rate) * tf.identity(self.lstm.Uog)
            self.bog = self.update_rate * self.bog + (1 - self.update_rate) * tf.identity(self.lstm.bog)

            self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.lstm.Wc)
            self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.lstm.Uc)
            self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def update_output_unit(self):
        self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits

        return unit
