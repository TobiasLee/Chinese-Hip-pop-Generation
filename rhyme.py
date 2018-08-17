from pypinyin import pinyin, FINALS, FINALS_TONE3
import numpy as np

Finals = ['i', 'u', 'v']


def n_rhyme(a, b):
    result = 0
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(a)):
        py1 = ''.join(a[i])
        py2 = ''.join(b[i])
        if py1[0] in Finals and len(py1) > 2:
            if py1[1] != 'n':
                py1 = py1[1:]
        if py2[0] in Finals and len(py2) > 2:
            if py2[1] != 'n':
                py2 = py2[1:]
        if py1 == py2:
            result = result + 1
        else:
            break
    return result


def rhyme(a, b):
    # 判断两句话是几押，返回0为不押韵
    # 两句话完全相同也返回0
    if a == b:
        return 0
    # N押 韵母和声调都要相同
    py1_tone = pinyin(a, style=FINALS_TONE3)
    py2_tone = pinyin(b, style=FINALS_TONE3)
    py1_tone.reverse()
    py2_tone.reverse()
    result = 0
    result = n_rhyme(py1_tone, py2_tone)
    if result > 1:
        return result
    # 单押和双押 韵母相同  声调可以不同
    py1 = pinyin(a, style=FINALS)[-2:]
    py2 = pinyin(b, style=FINALS)[-2:]
    py1.reverse()
    py2.reverse()
    result = n_rhyme(py1, py2)
    return result


# index -> sentence
def token2word(token, idx2word, reverse):
    s = []
    for tok in token:
        # print(tok)
        if tok != 0:
            # if tok != '0' and tok != ' ' and tok != '\n':
            try:
                s.append(idx2word[int(tok)])
            except KeyError:
                s.append('<UNK>')
    if reverse:
        return "".join(s)[::-1]
    else:
        return "".join(s)


def rhyme_reward(x, y, idx2word, reverse):
    return rhyme(token2word(x, idx2word, reverse), token2word(y, idx2word, reverse))


# 倒着试一下
def calc_rhyme(x_batch, y_batch, idx2word, reverse=False):
    reward = []
    for x, y in zip(x_batch, y_batch):
        reward.append(rhyme_reward(x, y, idx2word, reverse))
    return np.asarray(reward).reshape(len(x_batch), -1)


if __name__ == '__main__':
    word2idx = np.load("./data/w2i.npy").item()
    idx2word = {v: k for k, v in zip(word2idx.keys(), word2idx.values())}
    x = []
    with open("./data/dev_idx_x.txt", "r") as f:
        for i in f:
            x.append(i)
    y = []
    with open("./data/dev_idx_y.txt", "r") as f:
        for i in f:
            y.append(i)

    # a = np.array([[1, 2, 3],
    #               [3, 4, 5]])
    # print(np.mean(a, axis=0))
    # print(a - np.mean(a, axis=0))
    # print(x[75])
    # print(idx2word[1])
    # print()
    # print(word2idx['老铁'])
    # print(idx2word[1])
    # print(idx2word[1080])
    # print(x[75])
    print(x[75])
    print(y[75])
    print(word2idx['真'])
    print(idx2word[313])
    generated = np.asarray([[2994, 11, 2, 939, 1, 313, 1080],
                 [2, 3, 4, 5]])
    inputs = np.asarray([[1096, 4, 813, 1, 313, 1080, 0, 0],
              [2, 3, 4, 5]])
    print(rhyme_reward(np.array([1096, 4, 813, 1, 313, 1080, 0, 0]), np.array([2994, 11, 2, 939, 1, 313, 1080]),
                       idx2word))

    rewards = np.asarray([[.1, .1, .1, .1],
                         [.2, .2, .2, .2]])
    rr = calc_rhyme(generated, inputs, idx2word)
    print(rewards + rr)
    # print(rhyme('孤独生孤独', '泛滥得想吐纵横交错的金属发酥'))
    # print(rhyme('孤独生孤独', '泛滥得想吐纵横交错的金属负熟'))
    # print(rhyme('军火走私犯被拿下武器全部充公', '然后它们又被卖给非洲或者是中东'))
    print(rhyme('卸下了面具的真老<UNK>', '搭着我肩膀的真老铁'))

    print(np.mean(np.asarray([[0], [0], [1]]), axis=0))
    # print(rhyme('孤独生孤独', '孤独生孤独'))
    # print(rhyme('面对贝爷你还能做到百般淡定', '我看一会去医院没人帮你买单看病'))
    # print(rhyme('因', '晕'))
