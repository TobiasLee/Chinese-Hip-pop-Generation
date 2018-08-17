import numpy as np


class Input_Data_loader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, x_file, y_file):
        self.token_stream = []
        self.target_stream = []
        with open(x_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                # print(parse_line)
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)
        # print(len(self.token_stream))
        with open(y_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                # print(parse_line)
                if len(parse_line) == 20:
                    self.target_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        # print(self.num_batch)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.target_stream = self.target_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.target_batch = np.split(np.array(self.target_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        input = self.sequence_batch[self.pointer]
        target = self.target_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return input, target

    def reset_pointer(self):
        self.pointer = 0

    def get_all(self):
        return np.array(self.token_stream)


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                # print(parse_line)
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


#
if __name__ == '__main__':
    loader = Input_Data_loader(8)
    loader.create_batches("lyric.txt")

    batch, y = loader.next_batch()
    # print(batch)
    # print(y)
    # print(loader.get_all())
    # print(len(loader.get_all()))

    # print(loader.num_batch)
