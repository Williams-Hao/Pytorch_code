import numpy as np
from collections import defaultdict
import jieba
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = {'window_size': 2,
            'hidden_size': 4,
            'epochs': 50,
            'lr': 0.01}


class WordData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = torch.FloatTensor(self.data[item][0])
        y = np.argmax(self.data[item][1])

        return x,y


class Word2Vec(nn.Module):
    def __init__(self, vocab_size):
        super(Word2Vec, self).__init__()
        self.hidden_size = settings['hidden_size']
        self.lr = settings['lr']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

        self.W1 = nn.Linear(vocab_size, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, x):
        h = self.W1(x)
        # print(f'h shape:{h.shape}, content:{h}') # h.shape(Batch_size, hidden_size)
        z = self.W2(h)
        # print(f'z shape:{z.shape}, content:{z}') # h.shape(Batch_size, vocab_size)

        return z

def generate_training_data(corpus, settings):
    '''
    :param settings: 超参数
    :param corpus: 语料库
    :return: 训练样本
    '''

    word_counts = defaultdict(int)  # 当字典中不存在时返回0
    for row in corpus:
        for word in jieba.lcut(row, cut_all=False):
            # for word in row.split(' '):
            word_counts[word] += 1

    print("word_counts:", word_counts)

    hidden_size = settings['hidden_size']
    window_size = settings['window_size']

    vocab_size = len(word_counts.keys())  # hidden_size:不重复单词数
    words_list = list(word_counts.keys())  # words_list:单词列表
    word_index = dict((word, i) for i, word in enumerate(words_list))  # {单词:索引}
    index_word = dict((i, word) for i, word in enumerate(words_list))  # {索引:单词}

    def word2onehot(word):
        """
        :param word: 单词
        :return: ont-hot
        """
        # global word_index
        # global hidden_size
        word_vec = [0 for i in range(0, vocab_size)]  # 生成hidden_size维度的全0向量
        index = word_index[word]  # 获得word所对应的索引
        # print(f'word:{word}, word_vec:{word_vec}, index:{index}')
        word_vec[index] = 1  # 对应位置位1
        return word_vec

    training_data = []
    for sentence in corpus:
        tmp_list = sentence.split(' ')  # 语句单词列表
        tmp_list = jieba.lcut(row, cut_all=False)  # 语句单词列表
        print(f'sentence:{sentence}, tmp_list:{tmp_list}')
        sent_len = len(tmp_list)  # 语句长度
        for i, word in enumerate(tmp_list):  # 依次访问语句中的词语
            w_target = word2onehot(tmp_list[i])  # 中心词ont-hot表示
            w_context = []
            w_context_raw = []  # 上下文
            for j in range(i - window_size, i + window_size + 1):
                if j != i and j <= sent_len - 1 and j >= 0:
                    # print(f'tmp_list[j]:{tmp_list[j]}, onehot:{self.word2onehot(tmp_list[j])}')
                    # w_context.append(word2onehot(tmp_list[j]))
                    # w_context_raw.append(tmp_list[j])
                    training_data.append([w_target, word2onehot(tmp_list[j])])
            # training_data.append([w_target, w_context])  # 对应了一个训练样本
            # print(f'target: {tmp_list[i]}, context:{w_context_raw}')

    print(f'vocab_size:{vocab_size}')
    print("word_counts:", word_counts.items())
    print("word_list:", words_list)
    print("word_index:", word_index)
    print("index_word:", index_word)

    print("train_data:")
    for d in training_data:
        print("target:{}, context:{}".format(d[0], d[1]))

    return training_data, vocab_size


def train(model, epochs, data, plot_every):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    data_loader = DataLoader(data, batch_size=32, num_workers=1)

    for epoch in range(epochs):
        for it, (x, y) in enumerate(data_loader):
            # print(f'x.shape:{x.shape}, content:{x}') # shape (Batch_size, vocab_size)
            # print(f'y.shape:{y.shape}, content:{y}') # shape (Batch_size)

            model.zero_grad()

            y_pred = model(x)
            # print(f'y_pred shape:{y_pred.shape}')

            loss = criterion(y_pred, y)

            # print(f'loss:{loss}')

            loss.backward()

            optimizer.step()
            if (1+it) % plot_every == 0:
                logger.info('epoch:{}\tstep: {}\t loss:{}'.format(epoch, it+1, loss))


if __name__ == '__main__':
    corpus = ['我爱北京天安门', '我爱你中国', '中国首都是北京','中国地大物博','北京市我们中国人最热爱的地方']

    train_data, vocab_size = generate_training_data(corpus, settings)
    print(f'train_data len:{len(train_data)}')
    data_loader = WordData(train_data)
    print(data_loader.__getitem__(1))
    print(data_loader.__len__())

    w2c = Word2Vec(vocab_size)
    train(w2c, 500, data_loader, 5)