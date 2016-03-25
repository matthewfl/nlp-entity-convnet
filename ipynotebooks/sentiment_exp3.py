class SentimentExp(object):

    def __init__(self, train_X, train_Y, wordvecs=wordvectors):
        self.train_X = train_X
        self.train_Y = train_Y
        self.wordvecs = wordvecs

        self.input_size = 10  # not used
        self.batch_size = 50

        self.learning_rate = .01
        self.momentum = .9

        #self.train_X_rep = np.array([[self.getRep(x)] for x in self.train_X])
        self.train_X_rep = np.array([wordvecs.tokenize(x) for x in self.train_X])

        self._setup()

    def getRep(self, sent):
        ret = []
        for i in xrange(self.input_size):
            if i < len(sent):
                ret.append(self.wordvecs[sent[i]])
            else:
                ret.append(np.zeros(self.wordvecs.vector_size))
        return np.matrix(ret).reshape((1, self.input_size, self.wordvecs.vector_size))

    def _setup(self):
        self.x_batch = T.imatrix('x')
        self.y_batch = T.ivector('y')

        self.input_l = lasagne.layers.InputLayer((None, 50))

        self.embedding_l = EmbeddingLayer(
            self.input_l,
            W=self.wordvecs.get_numpy_matrix(),
            add_word_params=False,
        )

        self.first_l = lasagne.layers.Conv2DLayer(
            self.embedding_l,
            num_filters=100,
            filter_size=(2, self.wordvecs.vector_size),
            name='conv1',
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        self.first_l_max = lasagne.layers.MaxPool2DLayer(
            self.first_l,
            name='maxing1',
            pool_size=(1,49)  # the number 9 should be 50-1 since that would mean it maxes over the whole input....
        )

        self.hidden1_l = lasagne.layers.DenseLayer(
            self.first_l_max,
            num_units=50,
            name='dens1',
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        self.hidden1_l_drop = lasagne.layers.DropoutLayer(
            self.hidden1_l,
            name='drop1',
            p=.25,
        )

        self.out_l = lasagne.layers.DenseLayer(
            self.hidden1_l_drop,
            num_units=1,
            name='dens2',
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        self.output = lasagne.layers.get_output(self.out_l, self.x_batch)

        self.loss_vec_old = (self.output.reshape((self.output.size,)) - self.y_batch) ** 2
        self.output_diff = T.neq((self.output.flatten() > .5),(self.y_batch > .5)).sum()
        self.loss_vec = lasagne.objectives.binary_crossentropy(T.clip(self.output.reshape((self.output.size,)), .01, .99), self.y_batch)

        self.all_params = lasagne.layers.get_all_params(self.out_l)

        self.updates = lasagne.updates.adagrad(self.loss_vec.mean(), self.all_params, .01)
        #self.updates = lasagne.updates.apply_momentum(self.updates_adagrad)

        self.train_func = theano.function(
            [self.x_batch, self.y_batch],
            [self.loss_vec.mean(), self.loss_vec],
            updates=self.updates,
        )

        self.loss_func = theano.function(
            [self.x_batch, self.y_batch],
            [self.loss_vec.sum(), self.loss_vec, self.output_diff],
        )

    def _make_zero(self):
        self.embedding_l.W.get_value(borrow=True)[0,:] = 0

    def train(self):
        for s in xrange(0, len(self.train_X_rep), self.batch_size):
            end = s + self.batch_size
            if end > len(self.train_X_rep):
                end = len(self.train_X_rep)
            X_vals = np.array(self.train_X_rep[s:end]).astype('int32')
            y_vals = np.array(self.train_Y[s:end]).astype('int32')
            loss, _ = self.train_func(X_vals, y_vals)
            self._make_zero()

    def test_loss(self, test_X, test_Y):
        test_X_rep = np.array([self.wordvecs.tokenize(x) for x in test_X])
        loss_sum = 0.0
        wrong = 0.0
        for s in xrange(0, len(test_X_rep), self.batch_size):
            end = s + self.batch_size
            if end > len(self.train_X_rep):
                end = len(self.train_X_rep)
            X_vals = np.array(self.train_X_rep[s:end]).astype('int32')
            y_vals = np.array(self.train_Y[s:end]).astype('int32')
            loss, _, output_diff = self.loss_func(X_vals, y_vals)
            wrong += output_diff
            loss_sum += loss
        return loss_sum / len(test_X_rep), wrong / len(test_X_rep)

experiment = SentimentExp(sentiment.train_X, sentiment.train_Y)
