from generator import Generator
from dataloader import Input_Data_loader
import random
from util import *

# parameter
EMB_DIM = 256  # embedding dimension
HIDDEN_DIM = 256  # hidden state dimension of lstm cell
SEQ_LENGTH = 20  # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 0  # supervise (maximum likelihood estimation) epochs
PRE_DIS_NUM = 0
SEED = 88
vocab_size = 11681  # max idx of lyric token
np.random.seed(SEED)
random.seed(SEED)
test_x = "./data/test_idx_x_r.txt"
test_y = "./data/test_idx_y_r.txt"
# path to load model
model_path = "./model/seq_gan/"

# Note: generated num need to be greater than BATCH_SIZE
generated_num = 64
BATCH_SIZE = 32


def generate_paragraph(sess, generator_model, batch_size, generated_num, output_file, data_loader, sentence_num=3):
    generated_samples = []
    data_loader.reset_pointer()
    # for l in range(sentence_num):
    for i in range(generated_num // batch_size):
        target, input_idx = data_loader.next_batch()
        paragraph = [input_idx]
        for l in range(sentence_num):
            # print(l)
            one_batch = generator_model.generate(sess, paragraph[l])
            paragraph.append(one_batch)
        paragraph = np.asarray(paragraph)
        paragraph = np.transpose(paragraph, (1, 0, 2))
        generated_samples.append(paragraph)

    with open(output_file, 'w') as fout:
        for batch in generated_samples:
            for lyrics in batch:
                for line in lyrics:
                    buffer = ' '.join([str(x) for x in line]) + '\n'
                    fout.write(buffer)
                fout.write('\n')
    return generated_samples


if __name__ == '__main__':
    # load rhyme table
    table = np.load("./data/_table.npy")
    G = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, table, mode='infer',
                  has_input=True)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    restore_model(G, sess, saver, model_path=model_path)
    test_loader = Input_Data_loader(BATCH_SIZE)
    test_loader.create_batches(test_x, test_y)
    print("generating...")
    # generating according to input
    ret = generate_paragraph(sess, G, BATCH_SIZE, generated_num, "generated_paragraph.txt", test_loader)
    print("finished")
    # input_x = input()
