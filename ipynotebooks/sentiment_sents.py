import random


class Sentiment(object):

    def __init__(self, fdir, positive='rt-polarity.pos', negative='rt-polarity.neg'):
        with open(fdir + positive, 'r') as f:
            self.positive = f.readlines()
        with open(fdir + negative, 'r') as f:
            self.negative = f.readlines()
        self._make_train_test()

    def _make_train_test(self):
        data = self._make_arr()
        test_size = len(data) // 5
        train, test = data[test_size:], data[:test_size]
        self.train_X, self.train_Y = self._split(train)
        self.test_X, self.test_Y = self._split(test)

    def _split(self, x):
        return map(lambda a: a[0], x), map(lambda a: a[1], x)

    def _make_arr(self):
        sents = self.positive + self.negative
        labels = [1] * len(self.positive) + [0] * len(self.negative)
        res = zip(sents, labels)
        random.shuffle(res)
        return res
