from discriminator import Discriminator
from generator import Generator
from dataloader import Dis_dataloader, Input_Data_loader
import random
from g_beta import G_beta
import time
from util import *

# Discriminator Parameters
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_emb_size = 32

# Generator Parameters
EMB_DIM = 256  # embedding dimension
HIDDEN_DIM = 256  # hidden state dimension of lstm cell
SEQ_LENGTH = 20  # sequence length
START_TOKEN = 0
SEED = 88  # Random Seed, Xuan-Xue Seed
BATCH_SIZE = 256
vocab_size = 11681  #

# Adversarial Training Parameters
TOTAL_BATCH = 5  # Total Adversarial Epochs
PRE_GEN_NUM = 0  # supervise (maximum likelihood estimation) epochs
PRE_DIS_NUM = 0
generated_num = 256
sample_time = 16  # for G_beta to get reward
num_class = 2  # 0 : fake data 1 : real data
RHYME_WEIGHT = 1  # RHYME reward weight (no longer needed in table rhyme version)

ADV_GEN_TIME = 5
GEN_VS_DIS_TIME = 5

# data file paths
x_file = "./data/train_idx_small_x_r.txt"
y_file = "./data/train_idx_small_y_r.txt"

dev_x = "./data/dev_idx_x_r.txt"
dev_y = "./data/dev_idx_y_r.txt"

test_x = "./data/test_idx_x_r.txt"
test_y = "./data/test_idx_y_r.txt"

dev_file = "./result/dev_ret"
test_file = "./result/test_ret"

negative_file = './generator_sample.txt'
word2idx_file = "./data/w2i.npy"

dev_num = 1000
test_num = 1000
# pre-train model path, if no pre-train, please set the path to be None
# pre_train_gen_path = "./model/pre_gen/"
# pre_train_dis_path = "./model/pre_dis/"
# if None, means no model to load
pre_train_gen_path = None
pre_train_dis_path = None


def main():
    # load rhyme table
    table = np.load("./data/table.npy")
    np.random.seed(SEED)
    random.seed(SEED)

    # data loader
    # gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    input_data_loader = Input_Data_loader(BATCH_SIZE)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    D = Discriminator(SEQ_LENGTH, num_class, vocab_size, dis_emb_size, dis_filter_sizes, dis_num_filters, 0.2)
    G = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, table, has_input=True)

    # avoid occupy all the memory of the GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # savers for different models
    saver_gen = tf.train.Saver()
    saver_dis = tf.train.Saver()
    saver_seqgan = tf.train.Saver()

    # gen_data_loader.create_batches(positive_file)
    input_data_loader.create_batches(x_file, y_file)
    log = open('./experiment-log.txt', 'w')
    #  pre-train generator
    if pre_train_gen_path:
        print("loading pretrain generator model...")
        log.write("loading pretrain generator model...")
        restore_model(G, sess, saver_gen, pre_train_gen_path)
        print("loaded")
    else:
        log.write('pre-training generator...\n')
        print('Start pre-training...')
        for epoch in range(PRE_GEN_NUM):
            s = time.time()
            # loss = pre_train_epoch(sess, G, gen_data_loader)
            loss = pre_train_epoch(sess, G, input_data_loader)
            print("Epoch ", epoch, " loss: ", loss)
            log.write("Epoch:\t" + str(epoch) + "\tloss:\t" + str(loss) + "\n")
            print("pre-train generator epoch time: ", time.time() - s, " s")
            best = 1000
            if loss < best:
                saver_gen.save(sess, "./model/pre_gen/pretrain_gen_best")
                best = loss
    dev_loader = Input_Data_loader(BATCH_SIZE)
    dev_loader.create_batches(dev_x, dev_y)

    if pre_train_dis_path:
        print("loading pretrain discriminator model...")
        log.write("loading pretrain discriminator model...")
        restore_model(D, sess, saver_dis, pre_train_dis_path)
        print("loaded")
    else:
        log.write('pre-training discriminator...\n')
        print("Start pre-train the discriminator")
        s = time.time()
        for epoch in range(PRE_DIS_NUM):
            # generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
            generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file, input_data_loader)
            # dis_data_loader.load_train_data(positive_file, negative_file)
            dis_data_loader.load_train_data(y_file, negative_file)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        D.input_x: x_batch,
                        D.input_y: y_batch,
                        D.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, acc = sess.run([D.train_op, D.accuracy], feed)
            print("Epoch ", epoch, " Accuracy: ", acc)
            log.write("Epoch:\t" + str(epoch) + "\tAccuracy:\t" + str(acc) + "\n")
            best = 0
            # if epoch % 20  == 0 or epoch == PRE_DIS_NUM -1:
            #     print("saving at epoch: ", epoch)
            #     saver_dis.save(sess, "./model/per_dis/pretrain_dis", global_step=epoch)
            if acc > best:
                saver_dis.save(sess, "./model/pre_dis/pretrain_dis_best")
                best = acc
        print("pre-train discriminator: ", time.time() - s, " s")

    g_beta = G_beta(G, update_rate=0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('Start adversarial training...\n')

    for total_batch in range(TOTAL_BATCH):
        s = time.time()
        for it in range(ADV_GEN_TIME):
            for i in range(input_data_loader.num_batch):
                input_x, target = input_data_loader.next_batch()
                samples = G.generate(sess, input_x)
                rewards = g_beta.get_reward(sess, samples, input_x, sample_time, D)
                avg = np.mean(np.sum(rewards, axis=1), axis=0) / SEQ_LENGTH
                print(" epoch : %d time : %di: %d avg %f" % (total_batch, it, i, avg))
                feed = {G.x: samples, G.rewards: rewards, G.inputs: input_x}
                _ = sess.run(G.g_update, feed_dict=feed)
        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            avg = np.mean(np.sum(rewards, axis=1), axis=0) / SEQ_LENGTH
            buffer = 'epoch:\t' + str(total_batch) + '\treward:\t' + str(avg) + '\n'
            print('total_batch: ', total_batch, 'average reward: ', avg)
            log.write(buffer)

            saver_seqgan.save(sess, "./model/seq_gan/seq_gan", global_step=total_batch)

        g_beta.update_params()

        # train the discriminator
        for it in range(ADV_GEN_TIME // GEN_VS_DIS_TIME):
            # generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
            generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file, input_data_loader)
            dis_data_loader.load_train_data(y_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for batch in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        D.input_x: x_batch,
                        D.input_y: y_batch,
                        D.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(D.train_op, feed_dict=feed)
        print("Adversarial Epoch consumed: ", time.time() - s, " s")

    # final generation
    print("Finished")
    log.close()
    # save model

    print("Training Finished, starting to generating test ")
    test_loader = Input_Data_loader(batch_size=BATCH_SIZE)
    test_loader.create_batches(test_x, test_y)

    generate_samples(sess, G, BATCH_SIZE, test_num, test_file + "_final.txt", test_loader)
    # saver = tf.train.Saver()
    # saver.save(sess, './seq-gan')


if __name__ == '__main__':
    main()
