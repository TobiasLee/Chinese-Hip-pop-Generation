import tensorflow as tf
import numpy as np


def restore_model(model, sess, saver, model_path):
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.restore(sess, saver, ckpt.model_checkpoint_path)
    else:
        print("model not exists")


def generate_samples(sess, generator_model, batch_size, generated_num, output_file, data_loader):
    generated_samples = []
    data_loader.reset_pointer()

    for i in range(generated_num // batch_size):
        input, target = data_loader.next_batch()
        one_batch = generator_model.generate(sess, input)
        # print("batch:  ", one_batch)
        generated_samples.extend(one_batch)
    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()
    # print(data_loader.batch_num)
    for it in range(data_loader.num_batch):
        input_x, target = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, input_x, target)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
