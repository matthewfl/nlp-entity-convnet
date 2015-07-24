import csv


class StanfordSentimentReader(object):

    def __init__(self, dname):
        with open(dname + '/datasetSplit.txt') as f:
            c = csv.reader(f)
            c.next()
            mapping = dict(c)
        with open(dname + '/datasetSentences.txt') as f:
            c = csv.reader(f, delimiter='\t')
            c.next()
            sentences = dict(c)
        with open(dname + '/sentiment_labels.txt') as f:
            c = csv.reader(f, delimiter='|')
            c.next()
            scores = dict(c)
        pairs = dict((k, ([],[])) for k in range(1,4))
        for k, v in mapping.iteritems():
            p = pairs[int(v)]
            p[0].append(sentences[k])
            p[1].append(float(scores[k]))
        self.train_X, self.train_Y = pairs[1]
        self.test_X, self.test_Y = pairs[2]
        self.dev_X, self.dev_Y = pairs[3]
