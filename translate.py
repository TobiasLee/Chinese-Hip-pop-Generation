import numpy as np
import argparse

word2idx = np.load("./data/w2i.npy")
word2idx = word2idx.item()
id2word = {k: v for v, k in zip(word2idx.keys(), word2idx.values())}


def translate(word_indexs):
    words = []
    for idx in word_indexs:
        idx = int(idx)
        word = id2word.get(idx)
        #         print(word)
        if word:
            words.append(id2word.get(idx))
        # else:
        #     words.append("<UNK>")
    return "".join(words)


def parse(flag):
    if flag.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False


def translate_file(file_name, reverse):
    translated = []
    with open(file_name, "r") as f:
        for l in f:
            l = l.split()
            if parse(reverse):
                l = l[::-1]

            line = [int(x) for x in l]
            # trans = translate(line)
            translated.append(translate(line))
        # print(line)
    return translated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file', type=str, help='index file')
    parser.add_argument('--refer_file', default=None, type=str, help='reference file')
    parser.add_argument('--input_file', default=None, type=str, help='input file')
    parser.add_argument('--reverse', default="False", type=str, help='whether the token is reversed')
    args = parser.parse_args()

    if args.refer_file and args.input_file:
        trans = translate_file(args.index_file, args.reverse)
        refer = translate_file(args.refer_file, args.reverse)
        input = translate_file(args.input_file, args.reverse)
        for t, r, i in zip(trans, refer, input):
            print("input: ", i)
            print("translate: ", t)
            print("refer: ", r)
    else:
        for i in translate_file(args.index_file, args.reverse):
            print(i)

# print(translate([58,59, 3,60, 17 ,5 ,8 ,5 ,1 ,61, 62, 3, 63, 7 ,9 ,64 ,18, 3 ,19, 3]))
# with open(translate_file, "r") as f:
#     for l in f:
#         print(translate(l))
