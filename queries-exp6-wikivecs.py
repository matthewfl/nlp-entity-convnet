
# coding: utf-8

# change it to not use the document when looking for the link quality, to see how much that is actually helping

# In[1]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[2]:

from theano import *
from lasagne.layers import InputLayer, get_output
import lasagne
import lasagne.layers
import theano.tensor as T
import theano
import numpy as np
from helpers import SimpleMaxingLayer, SimpleAverageLayer
from wordvecs import WordVectors, EmbeddingLayer
import json
import re

theano.config.floatX = 'float32'
#theano.config.linker = 'cvm_nogc'
theano.config.openmp = True
theano.config.openmp_elemwise_minsize = 20000


# In[3]:

with open('/data/matthew/external-wiki2.json') as f:
    queries = json.load(f)['queries']

wordvectors = WordVectors(
    fname="/data/matthew/enwiki-20141208-pages-articles-multistream-links5-output1.bin",
    redir_fname='/data/matthew/enwiki-20141208-pages-articles-multistream-redirects5.json',
    negvectors=False,
    sentence_length=200,
)
wordvectors.add_unknown_words = False

page_redirects = wordvectors.redirects


from wikireader import WikiRegexes, WikipediaReader


def PreProcessedQueries(wikipedia_dump_fname, wordvec=wordvectors, queries=queries, redirects=page_redirects):

    get_words = re.compile('[^a-zA-Z0-9 ]')
    get_link = re.compile('.*?\[(.*?)\].*?')

    queried_pages = set()
    for docs, q in queries.iteritems():
        wordvec.tokenize(docs, length=200)
        for sur, v in q.iteritems():
            wrds_sur = get_words.sub(' ', sur)
            wordvec.tokenize(wrds_sur)
            link_sur = get_link.match(sur).group(1)
            wordvec.tokenize(link_sur)
            for link in v['vals'].keys():
                wrds = get_words.sub(' ', link)
                wordvec.tokenize(wrds)
                tt = WikiRegexes.convertToTitle(link)
                wordvec.get_location(tt)
                queried_pages.add(tt)

    added_pages = set()
    for title in queried_pages:
        if title in redirects:
            #wordvec.tokenize(self.redirects[title])
            added_pages.add(redirects[title])
    queried_pages |= added_pages

    page_content = {}

#     class GetWikipediaWords(WikipediaReader, WikiRegexes):

#         def readPage(ss, title, content):
#             tt = ss.convertToTitle(title)
#             if tt in queried_pages:
#                 cnt = ss._wikiToText(content)
#                 page_content[tt] = wordvec.tokenize(cnt)

#     GetWikipediaWords(wikipedia_dump_fname).read()

    rr = redirects
    rq = queried_pages
    rc = page_content

    class PreProcessedQueriesCls(object):

        wordvecs = wordvec
        queries = queries
        redirects = rr
        queried_pages = rq
        page_content = rc


    return PreProcessedQueriesCls


basePreProcessedQueries = PreProcessedQueries('/data/matthew/enwiki-20141208-pages-articles-multistream.xml')


class EntityVectorLinkExp(basePreProcessedQueries):

    batch_size = 1000 #20000
    num_training_items = 500000 #200000

    def __init__(self):
        self.sentence_length = self.wordvecs.sentence_length
        self.document_length = 100
        self.num_words_to_use_conv = 5

        self._setup()


    def _setup(self):
        #self.x_document_input = T.imatrix('x_doc')

        #self.x_document_id = T.ivector('x_doc_id')
        self.x_surface_text_input = T.imatrix('x_surface_link')
        self.x_surface_context_input = T.imatrix('x_surface_cxt')  # TODO

        self.x_target_input = T.ivector('x_target')
        self.x_target_words = T.imatrix('x_target_words')
        self.x_matches_surface = T.ivector('x_match_surface')
        self.x_link_id = T.ivector('x_link_id')

        #self.y_score = T.vector('y')
        self.y_answer = T.ivector('y_ans')  # contains the location of the gold answer so we can compute the loss
        self.y_grouping = T.imatrix('y_grouping')


        self.embedding_W = theano.shared(self.wordvecs.get_numpy_matrix().astype(theano.config.floatX))

#         self.document_l = lasagne.layers.InputLayer(
#             (None,self.document_length),
#             input_var=self.x_document_input
#         )

#         self.document_embedding_l = EmbeddingLayer(
#             self.document_l,
#             W=self.embedding_W,
#             add_word_params=False,
#         )

#         self.document_conv1_l = lasagne.layers.Conv2DLayer(
#             self.document_embedding_l,
#             num_filters=500,
#             filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
#             name='document_conv1',
#             nonlinearity=lasagne.nonlinearities.tanh,
#         )

#         self.document_max_l = lasagne.layers.Pool2DLayer(
#             self.document_conv1_l,
#             name='document_pool1',
#             pool_size=(self.document_length - self.num_words_to_use_conv, 1),
#             mode='sum',
#         )

#         self.document_dens1 = lasagne.layers.DenseLayer(
#             self.document_max_l,
#             num_units=250,
#             name='doucment_dens1',
#             nonlinearity=lasagne.nonlinearities.tanh,
#         )

#         self.document_drop1 = lasagne.layers.DropoutLayer(
#             self.document_dens1,
#             p=.25,
#         )

#         document_output_length = 200

#         self.document_dens2 = lasagne.layers.DenseLayer(
#             self.document_drop1,
#             num_units=225,
#             name='document_dens2',
#             nonlinearity=lasagne.nonlinearities.tanh,
#         )

#         self.document_drop2 = lasagne.layers.DropoutLayer(
#             self.document_dens2,
#             p=.25,
#         )

#         self.document_dens3 = lasagne.layers.DenseLayer(
#             self.document_drop2,
#             num_units=document_output_length,
#             name='document_dens3',
#             nonlinearity=lasagne.nonlinearities.tanh,
#         )

#         self.document_output = lasagne.layers.get_output(self.document_dens3)

        self.surface_context_l = lasagne.layers.InputLayer(
            (None, self.sentence_length),
            input_var=self.x_surface_context_input,
        )

        self.surface_context_embedding_l = EmbeddingLayer(
            self.surface_context_l,
            W=self.embedding_W,
            add_word_params=False,
        )

        self.surface_context_conv1_l = lasagne.layers.Conv2DLayer(
            self.surface_context_embedding_l,
            num_filters=300,
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            name='surface_cxt_conv1',
            nonlinearity=lasagne.nonlinearities.tanh,
        )

#         self.surface_context_avg1_l = SimpleAverageLayer(
#             [self.surface_context_conv1_l, self.surface_context_l],
#             #name='surface_context_avg'
#         )

        self.surface_context_pool1_l = lasagne.layers.Pool2DLayer(
            self.surface_context_conv1_l,
            name='surface_cxt_pool1',
            pool_size=(self.sentence_length - self.num_words_to_use_conv, 1),
            mode='sum',
        )

        self.surface_input_l = lasagne.layers.InputLayer(
            (None, self.sentence_length),
            input_var=self.x_surface_text_input
        )

        self.surface_embedding_l = EmbeddingLayer(
            self.surface_input_l,
            W=self.embedding_W,
            add_word_params=False,
        )

        self.surface_conv1_l = lasagne.layers.Conv2DLayer(
            self.surface_embedding_l,
            num_filters=300,
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            name='surface_conv1',
            nonlinearity=lasagne.nonlinearities.tanh,
        )

#         self.surface_avg1_l = SimpleAverageLayer(
#             [self.surface_conv1_l, self.surface_input_l],
#             #name='surface_avg'
#         )

        self.surface_pool1_l = lasagne.layers.Pool2DLayer(
            self.surface_conv1_l,
            name='surface_pool1',
            pool_size=(self.sentence_length - self.num_words_to_use_conv, 1),
            mode='sum',
        )

        self.surface_merged_l = lasagne.layers.ConcatLayer(
            [self.surface_context_pool1_l, self.surface_pool1_l]
        )

        self.surface_dens1 = lasagne.layers.DenseLayer(
            self.surface_merged_l,
            name='surface_dens1',
            num_units=250,
            nonlinearity=lasagne.nonlinearities.tanh,
        )

#         self.surface_drop1 = lasagne.layers.DropoutLayer(
#             self.surface_dens1,
#             p=.25,
#         )

#         self.surface_dens2 = lasagne.layers.DenseLayer(
#             self.surface_drop1,
#             name='surface_dens2',
#             num_units=200,
#             nonlinearity=lasagne.nonlinearities.tanh,
#         )

#         self.document_aligned_l = InputLayer(
#             (None, document_output_length),
#             input_var=self.document_output[self.x_document_id,:]
#         )

        ##############################################
        ## changed to not use the documented

#         self.source_l = lasagne.layers.ConcatLayer(
#             [self.document_aligned_l, self.surface_dens1]
#         )

        self.source_dens1 = lasagne.layers.DenseLayer(
            self.surface_dens1,   # CHANGED
            num_units=300,
            name='source_dens1',
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        self.source_drop1 = lasagne.layers.DropoutLayer(
            self.source_dens1,
            p=.25,
        )

        self.source_dens12 = lasagne.layers.DenseLayer(
            self.source_drop1,
            num_units=250,
            name='source_dens12',
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        self.source_drop12 = lasagne.layers.DropoutLayer(
            self.source_dens12,
            p=.25,
        )

        compared_vector_size = self.wordvecs.vector_size #+ 2 # extra space for if it matches the surface text

        self.source_dens2 = lasagne.layers.DenseLayer(
            self.source_drop12,
            num_units=compared_vector_size,  # this is the same size as the learned wikipedia vectors
            name='source_dens2',
            nonlinearity=lasagne.nonlinearities.linear,
        )

        self.source_out = lasagne.layers.get_output(self.source_dens2)

        matched_surface_reshaped = self.x_matches_surface.reshape(
            (self.x_matches_surface.shape[0], 1, 1, 1)).astype(theano.config.floatX)

        self.target_input_l = lasagne.layers.InputLayer(
            (None,),
            input_var=self.x_target_input
        )

        self.target_matched_surface_input_l = lasagne.layers.InputLayer(
            (None,1,1,1),
            input_var=matched_surface_reshaped,
        )

        self.target_embedding_l = EmbeddingLayer(
            lasagne.layers.reshape(self.target_input_l, ([0], 1)),
            W=self.embedding_W,
            add_word_params=False,
        )

        self.target_combined_feats_l = lasagne.layers.ConcatLayer(
            [self.target_embedding_l, self.target_matched_surface_input_l],
            axis=3
        )

        self.target_words_input_l = lasagne.layers.InputLayer(
            (None,self.sentence_length),
            input_var=self.x_target_words,
        )

        self.target_words_embedding_l = EmbeddingLayer(
            self.target_words_input_l,
            W=self.embedding_W,
            add_word_params=False,
        )

        self.target_words_conv1_l = lasagne.layers.Conv2DLayer(
            self.target_words_embedding_l,
            name='target_wrds_conv1',
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            num_filters=350,
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        self.target_words_pool1_l = lasagne.layers.Pool2DLayer(
            self.target_words_conv1_l,
            name='target_wrds_pool1',
            pool_size=(self.sentence_length - self.num_words_to_use_conv, 1),
            mode='sum',
        )

        self.target_merge_l = lasagne.layers.ConcatLayer(
            [lasagne.layers.reshape(self.target_words_pool1_l, ([0], [1])),
             lasagne.layers.reshape(self.target_embedding_l, ([0], [3]))]
        )

        self.target_dens1 = lasagne.layers.DenseLayer(
            self.target_merge_l,
            name='target_wrds_dens1',
            num_units=400,
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        self.target_drop1 = lasagne.layers.DropoutLayer(
            self.target_dens1,
            p=.25,
        )

        self.target_dens2 = lasagne.layers.DenseLayer(
            self.target_drop1,
            name='target_wrds_dens1',
            num_units=compared_vector_size,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        self.target_simple = lasagne.layers.DenseLayer(
            self.target_combined_feats_l,
            name='target_simple1',
            num_units=compared_vector_size,
            nonlinearity=lasagne.nonlinearities.linear,
        )

#         self.target_dens1 = lasagne.layers.DenseLayer(
#             self.target_conv1_l,
#             name='target_dens1',
#             num_units=300,
#             nonlinearity=lasagne.nonlinearities.tanh,
#         )

#         self.target_drop1 = lasagne.layers.DropoutLayer(
#             self.target_dens1,
#             p=.25,
#         )

#         self.target_dens2 = lasagne.layers.DenseLayer(
#             self.target_drop1,
#             name='target_dens2',
#             num_units=300,
#             nonlinearity=lasagne.nonlinearities.tanh,
#         )



        #self.target_out = lasagne.layers.get_output(self.target_embedding_l)


#         self.target_out = T.concatenate(
#             [self.embedding_W[self.x_target_input],
#              matched_surface_reshaped,
#             1-matched_surface_reshaped],
#              axis=1)


        #self.target_out = self.embedding_W[self.x_target_input]
        #self.target_out = lasagne.layers.get_output(self.target_dens2)

        self.target_out = lasagne.layers.get_output(self.target_simple)

        # compute the cosine distance between the two layers
        self.source_aligned_l = self.source_out[self.x_link_id, :]

        # this uses scan internally, which means that it comes back into python code to run the loop.....fml
        self.dotted_vectors =  T.batched_dot(self.target_out, self.source_aligned_l)
        # diag also does not support a C version.........
        #self.dotted_vectors = T.dot(self.target_out, self.source_aligned_l.T).diagonal()

        def augNorm(v):
            return T.maximum(T.basic.pow(T.basic.pow(T.basic.abs_(v), 2).sum(axis=1) + .001, .5), .001)

        self.res_l = self.dotted_vectors / (augNorm(self.target_out) * augNorm(self.source_aligned_l) + .001)
#         self.res_l = self.dotted_vectors / ((self.target_out.norm(1, axis=1) + .001) *
#                                             (self.source_aligned_l.norm(1, axis=1) + .001))

        self.res_cap = T.clip((T.tanh(self.res_l) + 1) / 2, .001, .999)

        #self.golds = self.res_cap[self.y_answer]

#         def maxOverRange(indx):
#             #return T.max(self.res_cap[T.arange(indx[0],indx[1])]) - self.res_cap[indx[2]]
#             #return -( self.res_l[indx[2]] - T.log(T.exp(self.res_l[T.arange(indx[0],indx[1])]).sum()) )
#             return -( self.res_l[indx[2]] - self.res_l[indx[0]])

#         # build a tensor to make a matrix with one set on each dimention
#         self.grouped, grouped_update = theano.scan(maxOverRange, sequences=self.y_grouping)

        def setSubSelector(indx, outputs):
            return T.set_subtensor(outputs[T.arange(indx[0], indx[1]), indx[3]], 1)

        num_target_samples = self.res_l.shape[0]

        select_seq = T.concatenate([
            self.y_grouping,
            T.arange(self.y_grouping.shape[0]).reshape((self.y_grouping.shape[0], 1))
        ], axis=1)

        self.selecting_matrix, _ = theano.scan(
            setSubSelector,
            outputs_info=T.zeros((num_target_samples, num_target_samples)),
            #n_steps=self.y_grouping.shape[0]
            sequences=select_seq,
        )

        self.groupped_elems = T.dot(self.selecting_matrix[-1], T.diag(T.exp(self.res_l)))
        self.groupped_res = T.log(self.groupped_elems.sum(axis=0)[T.arange(self.y_grouping.shape[0])])
        self.loss_vec = self.groupped_res - self.res_l[self.y_grouping[:,2]]

        self.all_params = (
            #lasagne.layers.get_all_params(self.target_dens2) +
            # TODO: add params for the target stuff,
            lasagne.layers.get_all_params(self.target_simple) +
            lasagne.layers.get_all_params(self.source_dens2)
            #lasagne.layers.get_all_params(self.document_dens2)
        )

        # weight the positive samples more since there are fewer of them,
        # freaking hack
        #self.loss_vec = -(10 * self.y_score * T.log(self.res_cap) + (1.0 - self.y_score) * T.log(1.0 - self.res_cap))

        #self.loss_vec = T.nnet.binary_crossentropy(self.res_cap, self.y_score)

        #self.loss_vec = T.exp(T.max(self.res_cap - self.res_cap[self.y_answer] + .1, 0)) - 1  # TODO: maybe have some squared term here or something?

        # this one works reasonably well
        #self.loss_vec = - T.log((T.clip(self.res_cap[self.y_answer] - self.res_cap, -1.0, 0.4) + 1.0) / 1.5)

        #self.loss_vec = self.grouped

        #self.loss_vec = - T.log((T.clip(self.res_l[self.y_answer] - self.res_l, -40.0, 10.0) + 40.0) / 51.0)
        #self.loss_vec = T.max(self.res_l[self.y_answer] - self.res_l + .1, 0)

        self.updates = lasagne.updates.adadelta(self.loss_vec.mean(), self.all_params)

        self.func_inputs = [
            #self.x_document_input,
            self.x_surface_text_input, self.x_surface_context_input, #self.x_document_id,
            self.x_target_input, self.x_matches_surface, self.x_link_id,
            self.y_answer, self.y_grouping
        ]  # self.x_target_words,

        self.train_func = theano.function(
            self.func_inputs,
            [self.res_cap, self.loss_vec.sum(), self.loss_vec],
            updates=self.updates,
            on_unused_input='ignore',
        )

        self.test_func = theano.function(
            self.func_inputs,
            [self.res_cap, self.loss_vec.sum(), self.loss_vec],
            on_unused_input='ignore',
        )

    def reset_accums(self):
        self.current_documents = []
        self.current_surface_context = []
        self.current_surface_link = []
        self.current_link_id = []
        self.current_target_input = []
        self.current_target_words = []
        self.current_target_matches_surface = []
        self.current_target_id = []
        self.current_target_goal = []
        self.current_learning_groups = []
        self.learning_targets = []

    def compute_batch(self, isTraining=True):
        if isTraining:
            func = self.train_func
        else:
            func = self.test_func
        self.reset_accums()
        self.total_links = 0
        self.total_loss = 0.0

        get_words = re.compile('[^a-zA-Z0-9 ]')
        get_link = re.compile('.*?\[(.*?)\].*?')

        for doc, queries in self.queries.iteritems():
            # skip the testing documents while training and vice versa
            if queries.values()[0]['training'] != isTraining:
                continue
            docid = len(self.current_documents)
            self.current_documents.append(self.wordvecs.tokenize(doc, length=self.document_length))
            for surtxt, targets in queries.iteritems():
                self.current_link_id.append(docid)
                surid = len(self.current_surface_link)
                self.current_surface_context.append(self.wordvecs.tokenize(get_words.sub(' ' , surtxt)))
                surlink = get_link.match(surtxt).group(1)
                self.current_surface_link.append(self.wordvecs.tokenize(surlink))
                surmatch = surlink.lower()
                #target_page_input = []
                target_words_input = []
                target_matches_surface = []
                target_inputs = []
                target_learings = []
                target_gold_loc = -1
                target_group_start = len(self.current_target_input)
                for target in targets['vals'].keys():
                    # skip the items that we don't know the gold for
                    if not targets['gold'] and isTraining:
                        continue
                    isGold = target == targets['gold']
                    #cnt = self.page_content.get(WikiRegexes.convertToTitle(target))
                    cnt = self.wordvecs.get_location(WikiRegexes.convertToTitle(target))
                    if cnt is None:
                        # were not able to find this wikipedia document
                        # so just ignore tihs result since trying to train on it will cause
                        # issues
                        continue
                    if isGold:
                        target_gold_loc = len(target_inputs)
                    #target_page_input.append(cnt)
                    target_words_input.append(self.wordvecs.tokenize(get_words.sub(' ', target)))
                    target_inputs.append(cnt)  # page_content already tokenized
                    target_matches_surface.append(int(surmatch == target.lower()))
                    target_learings.append((targets, target))
                if target_gold_loc is not None or not isTraining:  # if we can't get the gold item
                    # contain the index of the gold item for these items, so it can be less then it
                    gold_loc = (len(self.current_target_goal) + target_gold_loc)
                    self.current_target_goal += [gold_loc] * len(target_inputs)
                    self.current_target_input += target_inputs
                    self.current_target_id += [surid] * len(target_inputs)
                    self.current_target_words += target_words_input   # TODO: add
                    self.current_target_matches_surface += target_matches_surface
                    target_group_end = len(self.current_target_input)
                    self.current_learning_groups.append(
                        [target_group_start, target_group_end,
                         gold_loc])

                #self.current_target_goal.append(isGold)
                self.learning_targets += target_learings
            if len(self.current_target_id) > self.batch_size:
                #return
                self.run_batch(func)
                if self.total_links > self.num_training_items:
                    return self.total_loss / self.total_links

        if len(self.current_target_id) > 0:
            self.run_batch(func)

        return self.total_loss / self.total_links

    def run_batch(self, func):
        res_vec, loss_sum, loss_vec = func(
            #self.current_documents,
            self.current_surface_link, self.current_surface_context, #self.current_link_id,
            self.current_target_input, self.current_target_matches_surface, self.current_target_id,
            self.current_target_goal, self.current_learning_groups,
            # self.current_target_words,
        )
        self.check_params()
        self.total_links += len(self.current_target_id)
        self.total_loss += loss_sum
        for i in xrange(len(res_vec)):
            # save the results from this pass
            l = self.learning_targets[i]
            l[0]['vals'][ l[1] ] = res_vec[i]
        self.reset_accums()

    def check_params(self):
        if any([np.isnan(v.get_value(borrow=True)).any() for v in self.all_params]):
            raise RuntimeError('nan in some of the parameters')



queries_exp = EntityVectorLinkExp()

def evalCurrentState(trainingData=True, numSamples=50000):
    all_measured = 0
    all_correct = 0
    all_trained = 0
    for qu in queries.values():
        for en in qu.values():
            if en['training'] != trainingData:
                continue
            for e in en:
                if en['gold']:
                    if all_trained > numSamples:
                        break
                    all_measured += 1
                    all_trained += len(en['vals'].values())
                    m = max(en['vals'].values())
                    if en['vals'][en['gold']] == m and m != 0:
                        all_correct += 1

    r = all_measured, float(all_correct) / all_measured
    print r
    return r


# In[17]:

import random
def augmentTrainingData():
    for quk in queries.keys():
        qu = queries[quk]
        for enk in qu.keys():
            en = qu[enk]
            if not en['gold']:
                del qu[enk]
        if not qu:
            del queries[quk]
    for qu in queries.values():
        training = random.random() > .15
        for en in qu.values():
            en['training'] = training
augmentTrainingData()

queries_exp.check_params()


queries_exp.num_training_items = 500000


# In[18]:

#get_ipython().magic(u'time print queries_exp.compute_batch(False)')


# In[21]:

get_ipython().magic(u'time print queries_exp.compute_batch()')


# In[25]:

#print evalCurrentState(False, 500000)


# In[26]:

#print evalCurrentState(True, 500000)


# In[22]:

exp_results = []

for i in xrange(6):
    res = (i, queries_exp.compute_batch())
    print res
    exp_results.append(res)
exp_results.append(('testing run', queries_exp.compute_batch(False)))
exp_results.append(('training state', evalCurrentState(True)))
exp_results.append(('testing state', evalCurrentState(False)))


# In[23]:

print exp_results
