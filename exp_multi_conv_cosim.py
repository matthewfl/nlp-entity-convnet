from theano import *
from lasagne.layers import InputLayer, get_output
import lasagne
import lasagne.layers
import theano.tensor as T
import theano
import numpy as np
from helpers import SimpleMaxingLayer, SimpleAverageLayer
from wordvecs import WordVectors, EmbeddingLayer, WordTokenizer
from wikireader import WikiRegexes
#import json
import re
import random
import sys

theano.config.floatX = 'float32'
#theano.config.linker = 'cvm_nogc'
theano.config.openmp = True
theano.config.openmp_elemwise_minsize = 20000

from __main__ import baseModel, featureNames as featuresNames


class EntityVectorLinkExp(baseModel):

    batch_size = 250 #20000
    num_training_items = 500000 #200000
    dim_compared_vec = 150  # 100

    def __init__(self):
        self.sentence_length = self.wordvecs.sentence_length
        self.sentence_length_short = 10
        self.document_length = 100

        self.num_words_to_use_conv = 5
        self.enable_boosting = False
        self.num_negative_target_samples = 0 #1
        #self.enable_match_surface = False
        #self.enable_link_counts = True
        self.enable_train_wordvecs = False
        self.enable_cap_boosting = True

        self.num_indicator_features = len(featuresNames)

        self.main_nl = lasagne.nonlinearities.softmax# leaky_rectify

        self.impossible_query = featuresNames.index('Impossible')

        self._setup()

    def _setup(self):

        self.all_params = []

        self.all_conv_results = []
        self.all_conv_pool_results = []
        self.all_conv_names = []

        self.x_document_input = T.imatrix('x_doc')  # words from the source document

        self.x_document_id = T.ivector('x_doc_id')  # index of which source doucment this is from
        self.x_surface_text_input = T.imatrix('x_surface_link')  # text of the surface link
        self.x_surface_context_input = T.imatrix('x_surface_cxt')  #  words surrounding the surface link

        self.x_target_input = T.ivector('x_target')  # id of the target vector
        self.x_target_words = T.imatrix('x_target_words')  # words from the target title link
        self.x_matches_surface = T.ivector('x_match_surface')  # indicator if the target title matches the surface
        self.x_matches_counts = T.imatrix('x_matches_counts')  # info about the link counts
        self.x_target_document_words = T.imatrix('x_target_document_words')  # words from the body of target document
        self.x_link_id = T.ivector('x_link_id')  # indx of what link to compare to in the matrix

        self.x_denotaiton_features = T.matrix('x_denotation_ind_feats', dtype='int8')  # the joint denotation query features
        self.x_query_featurs = T.matrix('x_query_ind_feats', dtype='int8')  # the query features
        self.x_query_link_id = T.ivector('x_match_query')  # the query that a denotation links to
        self.x_denotation_ranges = T.imatrix('x_denotation_ranges')  # the range of joint denotations to sum over

        self.x_target_link_id = T.ivector('x_match_target')  # the target document that maches with a given denotation

        self.y_isgold = T.vector('y_gold', dtype='int8')  # is 1 if the gold item, 0 otherwise
        self.y_grouping = T.imatrix('y_grouping')  # matrix containing [start_idx, end_idx, gold_idx]

        self.embedding_W = theano.shared(self.wordvecs.get_numpy_matrix().astype(theano.config.floatX),name='embedding_W')
        self.embedding_W_docs = theano.shared(self.documentvecs.get_numpy_matrix().astype(theano.config.floatX),name='embedding_W_docs')

        def augRectify(x):
            # if x is zero, then the gradient failes due to computation: x / |x|
            return T.maximum(x, -.01 * x)

        simpleConvNonLin = augRectify

        self.document_l = lasagne.layers.InputLayer(
            (None,self.document_length),
            input_var=self.x_document_input
        )

        self.document_embedding_l = EmbeddingLayer(
            self.document_l,
            W=self.embedding_W,
            add_word_params=self.enable_train_wordvecs,
        )

        self.document_simple_conv1_l = lasagne.layers.Conv2DLayer(
            self.document_embedding_l,
            num_filters=self.dim_compared_vec,
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            name='document_simple_conv',
            nonlinearity=simpleConvNonLin,
        )

        self.all_conv_names.append('document_conv')
        self.all_conv_results.append(lasagne.layers.get_output(self.document_simple_conv1_l))

        self.document_simple_sum_l = lasagne.layers.Pool2DLayer(
            self.document_simple_conv1_l,
            name='document_simple_pool',
            pool_size=(self.document_length - self.num_words_to_use_conv, 1),
            mode='sum',
        )

        self.all_conv_pool_results.append(lasagne.layers.get_output(self.document_simple_sum_l))

        self.document_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.document_simple_sum_l, ([0],-1)))

        self.all_params += lasagne.layers.get_all_params(self.document_simple_sum_l)


        ##########################################
        ## surface text

        self.surface_context_l = lasagne.layers.InputLayer(
            (None, self.sentence_length),
            input_var=self.x_surface_context_input,
        )

        self.surface_context_embedding_l = EmbeddingLayer(
            self.surface_context_l,
            W=self.embedding_W,
            add_word_params=self.enable_train_wordvecs,
        )

        self.surface_context_conv1_l = lasagne.layers.Conv2DLayer(
            self.surface_context_embedding_l,
            num_filters=self.dim_compared_vec,
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            name='surface_cxt_conv1',
            nonlinearity=simpleConvNonLin,
        )

        self.all_conv_names.append('surface_context_conv')
        self.all_conv_results.append(lasagne.layers.get_output(self.surface_context_conv1_l))

        self.surface_context_pool1_l = lasagne.layers.Pool2DLayer(
            self.surface_context_conv1_l,
            name='surface_cxt_pool1',
            pool_size=(self.sentence_length - self.num_words_to_use_conv, 1),
            mode='sum',   # WAS 'MAX' FOR SOME REASON
        )

        self.all_conv_pool_results.append(lasagne.layers.get_output(self.surface_context_pool1_l))

        self.surface_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.surface_context_pool1_l, ([0], -1))
        )

        self.all_params += lasagne.layers.get_all_params(self.surface_context_pool1_l)

        self.surface_input_l = lasagne.layers.InputLayer(
            (None, self.sentence_length_short),
            input_var=self.x_surface_text_input
        )

        self.surface_embedding_l = EmbeddingLayer(
            self.surface_input_l,
            W=self.embedding_W,
            add_word_params=self.enable_train_wordvecs,
        )

        self.surface_conv1_l = lasagne.layers.Conv2DLayer(
            self.surface_embedding_l,
            num_filters=self.dim_compared_vec,
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            name='surface_conv1',
            nonlinearity=simpleConvNonLin,
        )

        self.all_conv_names.append('surface_conv')
        self.all_conv_results.append(lasagne.layers.get_output(self.surface_conv1_l))

        self.surface_pool1_l = lasagne.layers.Pool2DLayer(
            self.surface_conv1_l,
            name='surface_pool1',
            pool_size=(self.sentence_length_short - self.num_words_to_use_conv, 1),
            mode='sum',
        )

        self.all_conv_pool_results.append(lasagne.layers.get_output(self.surface_pool1_l))

        self.surface_words_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.surface_pool1_l, ([0], -1))
        )

        self.all_params += lasagne.layers.get_all_params(self.surface_pool1_l)


        ###################################################
        ## dealing with the target side

        # matched_surface_reshaped = self.x_matches_surface.reshape(
        #     (self.x_matches_surface.shape[0], 1, 1, 1)).astype(theano.config.floatX)

        self.target_input_l = lasagne.layers.InputLayer(
            (None,),
            input_var=self.x_target_input
        )

        #################################
        ## target indicators features

        ## these have been replaced with the indicatores as provided by the scala system
        # self.target_matched_surface_input_l = lasagne.layers.InputLayer(
        #     (None,1,1,1),
        #     input_var=matched_surface_reshaped,
        # )

        # self.target_matched_counts_input_l = lasagne.layers.InputLayer(
        #     (None,5),
        #     input_var=self.x_matches_counts.astype(theano.config.floatX),
        # )

        # words from the title of the target
        self.target_words_input_l = lasagne.layers.InputLayer(
            (None,self.sentence_length_short),
            input_var=self.x_target_words,
        )

        self.target_words_embedding_l = EmbeddingLayer(
            self.target_words_input_l,
            W=self.embedding_W,
            add_word_params=self.enable_train_wordvecs,
        )

        self.target_words_conv1_l = lasagne.layers.Conv2DLayer(
            self.target_words_embedding_l,
            name='target_wrds_conv1',
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            num_filters=self.dim_compared_vec,
            nonlinearity=simpleConvNonLin,
        )

        self.all_conv_names.append('target_title_conv')
        self.all_conv_results.append(lasagne.layers.get_output(self.target_words_conv1_l))

        self.target_words_pool1_l = lasagne.layers.Pool2DLayer(
            self.target_words_conv1_l,
            name='target_wrds_pool1',
            pool_size=(self.sentence_length_short - self.num_words_to_use_conv, 1),
            mode='sum',
        )

        self.all_conv_pool_results.append(lasagne.layers.get_output(self.target_words_pool1_l))

        self.target_title_out = lasagne.layers.get_output(
            lasagne.layers.reshape(self.target_words_pool1_l, ([0],-1))
        )

        self.all_params += lasagne.layers.get_all_params(self.target_words_pool1_l)


        # words from the body of the target
        self.target_body_words_input_l = lasagne.layers.InputLayer(
            (None,self.sentence_length),
            input_var=self.x_target_document_words,
        )

        self.target_body_words_embedding_l = EmbeddingLayer(
            self.target_body_words_input_l,
            W=self.embedding_W,
            add_word_params=self.enable_train_wordvecs,
        )

        self.target_body_simple_conv1_l = lasagne.layers.Conv2DLayer(
            self.target_body_words_embedding_l,
            name='target_body_simple_conv',
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            num_filters=self.dim_compared_vec,
            nonlinearity=simpleConvNonLin,
        )

        self.all_conv_names.append('target_body_conv')
        self.all_conv_results.append(lasagne.layers.get_output(self.target_body_simple_conv1_l))

        self.target_body_simple_sum_l = lasagne.layers.Pool2DLayer(
            self.target_body_simple_conv1_l,
            name='target_body_simple_sum',
            pool_size=(self.sentence_length - self.num_words_to_use_conv, 1),
            mode='sum',
        )

        self.all_conv_pool_results.append(lasagne.layers.get_output(self.target_body_simple_sum_l))

        self.target_out = lasagne.layers.get_output(
            lasagne.layers.reshape(self.target_body_simple_sum_l, ([0],-1)))

        self.all_params += lasagne.layers.get_all_params(self.target_body_simple_sum_l)

        #########################################################
        ## compute the cosine distance between the two layers

        # the are going to multiple entity links per document so we have the `_id` ivectors that represent how
        # we need to reshuffle the inputs, this saves on computation

        # source body
        self.source_aligned_l = self.document_output[self.x_document_id,:][self.x_link_id,:]
        # source context
        self.source_context_aligned_l = self.surface_output[self.x_link_id,:]
        # source surface words
        self.source_surface_words_aligned_l = self.surface_words_output[self.x_link_id,:]


        def augNorm(v):
            return T.basic.pow(T.basic.pow(T.basic.abs_(v), 2).sum(axis=1) + .001, .5)

        def cosinsim(a, b):
            dotted = T.batched_dot(a, b)
            return dotted / (augNorm(a) * augNorm(b))

        def comparedVLayers(a, b):
            dv = cosinsim(a, b)
            return lasagne.layers.InputLayer(
                (None,1),
                input_var=dv.reshape((dv.shape[0], 1))
            )

        self.cosine_combined = lasagne.layers.concat(
            [
               comparedVLayers(self.target_out, self.source_aligned_l),
               comparedVLayers(self.target_out, self.source_context_aligned_l),
               comparedVLayers(self.target_out, self.source_surface_words_aligned_l),

               comparedVLayers(self.target_title_out, self.source_aligned_l),
               comparedVLayers(self.target_title_out, self.source_context_aligned_l),
               comparedVLayers(self.target_title_out, self.source_surface_words_aligned_l),
            ],
            axis=1
        )

        self.cosine_weighted = lasagne.layers.DenseLayer(
            self.cosine_combined,
            name='cosine_dens1',
            num_units=1,
            b=None,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        self.cosine_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.cosine_weighted, (-1,)))

        self.all_params += lasagne.layers.get_all_params(self.cosine_weighted)


        ######################################################
        ## indicator feature input


        self.query_feat_l = lasagne.layers.InputLayer(
            (None,self.num_indicator_features),
            input_var=self.x_query_featurs,
        )

        #rank_feats = [f[0] for f in enumerate(featuresNames) if f[1].startswith('Rank=')]

        self.denotation_join_feat_l = lasagne.layers.InputLayer(
            (None,self.num_indicator_features),
            input_var=self.x_denotaiton_features,#[:, rank_feats],
        )

        ## the query and denotation features are now combined when inputed into the same denotation vector

        # self.query_layer_l = lasagne.layers.DenseLayer(
        #     self.query_feat_l,
        #     name='query_lin',
        #     num_units=1,
        #     nonlinearity=lasagne.nonlinearities.linear,
        # )

        # self.query_output = lasagne.layers.get_output(
        #     lasagne.layers.reshape(self.query_layer_l, (-1,))
        # )

        # self.all_params += lasagne.layers.get_all_params(self.query_layer_l)

        # self.aligned_queries = self.query_output[self.x_query_link_id]

        self.aligned_cosine = self.cosine_output[self.x_target_link_id]

        self.denotation_layer_l = lasagne.layers.DenseLayer(
            self.denotation_join_feat_l,
            name='denotation_lin',
            num_units=1,
            nonlinearity=lasagne.nonlinearities.linear,
            #W=self.query_layer_l.W,
        )

        self.denotation_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.denotation_layer_l, (-1,)))

        self.all_params += lasagne.layers.get_all_params(self.denotation_layer_l)

        ###########################
        ## multiply the two parts of the join scores

        self.unmerged_scores =  (
            ( #(self.aligned_queries) +
            (self.denotation_output))
            + self.aligned_cosine
        )

        #############################################
        ## normalizing the scores and recombining
        ## the output if there were multiple entries
        ## for the same target document
        #############################################


        def sloppyMathLogSum(vals):
            m = vals.max()
            return T.log(T.exp(vals - m).sum()) + m

        def mergingSum(indx, unmerged):
            return sloppyMathLogSum(unmerged[T.arange(indx[0], indx[1])])

        self.merged_scores, _ = theano.scan(
            mergingSum,
            sequences=[self.x_denotation_ranges],
            non_sequences=[self.unmerged_scores]
        )

        ########################################
        ## true output values
        ########################################

        self.unscaled_output = self.merged_scores

        def scaleRes(indx, outputs, res):
            ran = T.arange(indx[0], indx[1])
            s = sloppyMathLogSum(res[ran])
            return T.set_subtensor(outputs[ran], res[ran] - s)

        self.scaled_scores, _ = theano.scan(
            scaleRes,
            sequences=[self.y_grouping],
            non_sequences=[self.unscaled_output],
            outputs_info=T.zeros((self.unscaled_output.shape[0],))
        )

        self.true_output =  self.scaled_scores[-1]

        ############################
        ## compute the loss
        ############################

        def lossSum(indx, res):
            return sloppyMathLogSum(res[T.arange(indx[0], indx[1])])

        self.groupped_res, _ = theano.scan(
            lossSum,
            sequences=[self.y_grouping],
            non_sequences=[self.true_output],
        )

        def selectGolds(indx, res, golds):
            r = T.arange(indx[0], indx[1])
            # fix some issue with theano?
            # the gold value should simply comes from the input
            # so there is no good reason to have to disconnect the graident here
            gs = theano.gradient.disconnected_grad(golds[r])
            vals = gs * res[r] + (1 - gs) * -1000000  # approx 0
            return sloppyMathLogSum(vals)

        self.gold_res, _ = theano.scan(
            selectGolds,
            sequences=[self.y_grouping],
            non_sequences=[self.true_output, self.y_isgold],
        )

        self.loss_vec = self.groupped_res - self.gold_res

        self.loss_scalar = self.loss_vec.sum()

        self.updates = lasagne.updates.adadelta(
            self.loss_scalar / self.loss_vec.shape[0] ,
            self.all_params)

        self.func_inputs = [
            self.x_document_input,
            self.x_surface_text_input, self.x_surface_context_input, self.x_document_id,
            self.x_target_input, self.x_matches_surface, self.x_matches_counts, self.x_link_id,
            self.x_target_words, self.x_target_document_words,
            self.x_denotaiton_features, self.x_query_featurs, self.x_query_link_id, self.x_denotation_ranges,
            self.x_target_link_id,
            self.y_grouping,
            self.y_isgold,
        ]

        self.func_outputs = [
            self.true_output,
            self.loss_vec.sum(),
            self.loss_scalar,
            self.loss_vec,
            #self.res_l,
        ]

        self.train_func = theano.function(
            self.func_inputs,
            self.func_outputs,
            updates=self.updates,
            on_unused_input='ignore',
        )

        self.test_func = theano.function(
            self.func_inputs,
            self.func_outputs,
            on_unused_input='ignore',
        )

        self.find_conv_active_func = theano.function(
            self.func_inputs,
            self.all_conv_results,
            on_unused_input='ignore',
        )

    def reset_accums(self):
        self.current_documents = []
        self.current_surface_context = []
        self.current_surface_link = []
        self.current_link_id = []
        self.current_target_input = []
        self.current_target_words = []
        self.current_target_body_words = []
        self.current_target_matches_surface = []
        self.current_target_id = []
        self.current_target_is_gold = []
#         self.current_target_goal = []
#         self.current_feat_indicators = []
        self.current_learning_groups = []
        self.learning_targets = []
        self.current_surface_target_counts = []
        #self.current_boosted_groups = []

        self.current_queries = []
        self.current_denotations_feats_indicators = []
        self.current_denotations_related_query = []
        self.current_denotations_range = []
        self.current_denotation_targets_linked = []

        self.failed_match = []

    def compute_batch(self, isTraining=True, useTrainingFunc=True, batch_run_func=None):
        if isTraining and useTrainingFunc:
            func = self.train_func
        else:
            func = self.test_func
        if batch_run_func is None:
            batch_run_func = self.run_batch
        self.reset_accums()
        self.total_links = 0
        self.total_loss = 0.0

        get_words = re.compile('[^a-zA-Z0-9 ]')
        get_link = re.compile('.*?\[(.*?)\].*?')

        empty_sentence = np.zeros(self.sentence_length, dtype='int32')

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
                self.current_surface_link.append(self.wordvecs.tokenize(surlink, length=self.sentence_length_short))
                surmatch = surlink.lower()
                surcounts = self.surface_counts.get(surmatch)
                if not surcounts:
                    self.failed_match.append(surmatch)
                    surcounts = {}
                target_body_words_input = []  # words from the target document
                target_words_input = []  # the words from the target title
                target_matches_surface = []
                target_inputs = []  # the target vector
                target_learings = []
                target_match_counts = []
                target_gold_loc = -1
                target_group_start = len(self.current_target_input)
#                 target_feat_indicators = []

                denotations_joint_indicators = []
                denotations_linked_query = []
                denotations_range = []

                denotation_target_linked = []

                target_isgold = []

                queries_feats_indicators = []
                for ind in targets['query_vals']:
                    query_feats = np.zeros((self.num_indicator_features,), dtype='int8')
#                     query_feats[ind] = 1
                    queries_feats_indicators.append(query_feats)
                queries_len = len(targets['query_vals'])

                for target in set(targets['vals'].keys() +
                                 random.sample(self.documentvecs.reverse_word_location, self.num_negative_target_samples)
                                  ) - {None,}:
                    isGold = target in targets['gold']
                    wiki_title = WikiRegexes.convertToTitle(target)
                    cnt_wrds = self.page_content.get(wiki_title) #WikiRegexes.convertToTitle(target))
                    cnt = self.documentvecs.get_location(wiki_title)
                    if wiki_title == 'nil':
                        cnt = 0  # this is the stop symbol location
                    if cnt is None:
                        # were not able to find this wikipedia document
                        # so just ignore tihs result since trying to train on it will cause
                        # issues
                        if cnt_wrds is None:
                            # really know nothing
                            continue
                        else:
                            # we must not have had enough links to this document
                            # but still have the target text
                            cnt = 0
                    if isGold:
                        target_gold_loc = len(target_inputs)
                        target_isgold.append(1)
                    else:
                        target_isgold.append(0)
                    target_body_words_input.append(cnt_wrds if cnt_wrds is not None else empty_sentence)
                    target_words_input.append(self.wordvecs.tokenize(get_words.sub(' ', target), length=self.sentence_length_short))
                    target_inputs.append(cnt)
                    # page_content already tokenized
                    target_matches_surface.append(int(surmatch == target.lower()))
                    target_learings.append((targets, target))
                    target_match_counts.append(surcounts.get(wiki_title, 0))

                    joint_indicators = []
                    query_idx = []
                    indicators_place = targets['vals'].get(target)
                    if indicators_place:
                        # [queries][indicator id]
                        for indx in xrange(len(indicators_place[1])):
                            local_feats = np.zeros((self.num_indicator_features,), dtype='int8')
                            local_feats[indicators_place[1][indx]] = 1
                            local_feats[targets['query_vals'][indx]] = 1  # features from the joint
#                             if isGold:  #################################### hack
#                                 local_feats[-1] = 1
                            joint_indicators.append(local_feats)
                            query_idx.append(len(self.current_queries) + indx)
                    else:
                        raise NotImplementedError()
                        for indx in xrange(queries_len):
                            local_feats = np.zeros((self.num_indicator_features,), dtype='int8')
                            local_feats[self.impossible_query] = 1
                            joint_indicators.append(local_feats)
                            query_idx.append(len(self.current_queries) + indx)

                    start_range = len(denotations_joint_indicators) + len(self.current_denotations_feats_indicators)
                    denotations_joint_indicators += joint_indicators
                    denotations_linked_query += query_idx
                    denotations_range.append([start_range, start_range + len(joint_indicators)])
                    denotation_target_linked += [len(self.current_target_words) + len(target_words_input) - 1] * len(query_idx)


#                     indicators = np.zeros((self.num_indicator_features,), dtype='int8')
#                     if indicators_place:

#                         indicators[indicators_place[1]] = 1
#                     target_feat_indicators.append(indicators)

                    #if wiki_title not in surcounts:
                    #    print surcounts, wiki_title
                if target_gold_loc is not None or not isTraining:  # if we can't get the gold item
                    # contain the index of the gold item for these items, so it can be less then it
#                     gold_loc = (len(self.current_target_goal) + target_gold_loc)
                    sorted_match_counts = [-4,-3,-2,-1] + sorted(set(target_match_counts))
                    #print sorted_match_counts
                    target_match_counts_indicators = [
                        [
                            int(s == sorted_match_counts[-1]),
                            int(s == sorted_match_counts[-2]),
                            int(s == sorted_match_counts[-3]),
                            int(0 < s <= sorted_match_counts[-4]),
                            int(s == 0),
                        ]
                        for s in target_match_counts
                    ]
#                     self.current_target_goal += [gold_loc] * len(target_inputs)
                    self.current_target_input += target_inputs
                    self.current_target_id += [surid] * len(target_inputs)
                    self.current_target_words += target_words_input
                    self.current_target_matches_surface += target_matches_surface
                    self.current_surface_target_counts += target_match_counts_indicators
                    self.current_target_body_words += target_body_words_input
#                     self.current_feat_indicators += target_feat_indicators
                    self.current_target_is_gold += target_isgold

                    target_group_end = len(self.current_target_input)
                    self.current_learning_groups.append(
                        [target_group_start, target_group_end,
                        -1 # gold_loc
                        ])
                    #self.current_boosted_groups.append(targets['boosted'])

                    self.current_queries += queries_feats_indicators

                    self.current_denotations_feats_indicators += denotations_joint_indicators
                    self.current_denotations_related_query += denotations_linked_query
                    self.current_denotations_range += denotations_range

                    self.current_denotation_targets_linked += denotation_target_linked

                #self.current_target_goal.append(isGold)
                self.learning_targets += target_learings
            if len(self.current_target_id) > self.batch_size:
#                 return
                batch_run_func(func)
                sys.stderr.write('%i\r'%self.total_links)
                if self.total_links > self.num_training_items:
                    return self.total_loss / self.total_links, #self.total_boosted_loss / self.total_links

        if len(self.current_target_id) > 0:
            batch_run_func(func)
            #self.run_batch(func)

        return self.total_loss / self.total_links, #self.total_boosted_loss / self.total_links

    def run_batch(self, func):
        res_vec, loss_sum, _, loss_vec, = func(
            self.current_documents,
            self.current_surface_link, self.current_surface_context, self.current_link_id,
            self.current_target_input, self.current_target_matches_surface, self.current_surface_target_counts, self.current_target_id,
            self.current_target_words, self.current_target_body_words, #self.current_feat_indicators,
            self.current_denotations_feats_indicators, self.current_queries, self.current_denotations_related_query, self.current_denotations_range,
            self.current_denotation_targets_linked,
            #self.current_target_goal,
            self.current_learning_groups, #self.current_boosted_groups,
            self.current_target_is_gold,
        )
        self.check_params()
        self.total_links += len(self.current_target_id)
        self.total_loss += loss_sum
        #self.total_boosted_loss += loss_boosted
        learned_groups = []  # right...dict not hashable....
        for i in xrange(len(res_vec)):
            # save the results from this pass
            l = self.learning_targets[i]
            if l[1] in l[0]['vals']:
                l[0]['vals'][ l[1] ][0] = float(res_vec[i]), 0#float(nn_outs[i])
            if l[0] not in learned_groups:
                learned_groups.append(l[0])
        self.reset_accums()

    def run_batch_max_activate(self, _):
        res = self.find_conv_active_func(
            self.current_documents,
            self.current_surface_link, self.current_surface_context, self.current_link_id,
            self.current_target_input, self.current_target_matches_surface, self.current_surface_target_counts, self.current_target_id,
            self.current_target_words, self.current_target_body_words,
            self.current_denotations_feats_indicators, self.current_queries, self.current_denotations_related_query, self.current_denotations_range,
            self.current_denotation_targets_linked,
            self.current_learning_groups,
            self.current_target_is_gold,
        )
        self.check_params()
        self.total_links += len(self.current_target_id)
        self.total_loss += 0

        # need to match with the conv names/results
        conv_inputs = [
            # shape: (document index, number of words)
            np.array(self.current_documents),
            np.array(self.current_surface_context),
            np.array(self.current_surface_link),
            np.array(self.current_target_words),
            np.array(self.current_target_body_words),
        ]

        # res shape: (document index, num filters, output rows, output columns [word vectors, should be 1])

        for i in xrange(len(res)):
            conv_len = conv_inputs[i].shape[1] - res[i].shape[2] + 1
            for dim in xrange(self.dim_compared_vec):
                current_min = self.conv_max[i][dim][0][0]
                higher_p = res[i][:, dim, :, 0] > current_min
                if higher_p.any():
                    higher_where = np.where(higher_p)
                    higher_vals = res[i][higher_where[0], dim, higher_where[1], 0]
                    current_words = set(w[1] for w in self.conv_max[i][dim])
                    higher_words = [
                        # np won't do this selecting in one shot
                        conv_inputs[i][higher_where[0][w], higher_where[1][w]:(higher_where[1][w]+conv_len)]
                        for w in xrange(len(higher_where[0]))
                    ]
                    #higher_words = conv_inputs[i][higher_where[0], higher_where[1]:(higher_where[1]+conv_len)]
                    itm_arr = self.conv_max[i][dim]
                    for x in xrange(len(higher_vals)):
                        hv = higher_vals[x]
                        if hv > itm_arr[0][0]:
                            # this is higher then the current min value
                            words = tuple(higher_words[x])
                            if words not in current_words:
                                # we should remove the min element
                                itm_arr[0] = (hv, words)
                                itm_arr.sort()
                                current_words = set(w[1] for w in itm_arr)
        
        self.reset_accums()


    def find_max_convs(self, num_per_activation=10):
        assert len(self.all_conv_names) == 5
        assert len(self.all_conv_results) == 5

        #per_act = [(0, ())] * num_per_activation
        #per_conv = [per_act] * self.dim_compared_vec
        #self.conv_max = [per_conv] * len(self.all_conv_names)

        self.conv_max = [
            [
                [
                    (0, ())
                    for c in xrange(num_per_activation)
                ]
                for b in xrange(self.dim_compared_vec)
            ]
            for a in xrange(len(self.all_conv_names))
        ]

        self.compute_batch(True, batch_run_func=self.run_batch_max_activate)
        self.compute_batch(False, batch_run_func=self.run_batch_max_activate)

        # convert the second part of the arrays to strings of words

        for ci in xrange(len(self.all_conv_names)):
            for di in xrange(self.dim_compared_vec):
                for ai in xrange(num_per_activation):
                    a = self.conv_max[ci][di][ai]
                    self.conv_max[ci][di][ai] = (a[0], ' '.join([str(self.wordvecs.get_word(w)) for w in a[1]]))

        return self.conv_max


    def check_params(self):
        if any([np.isnan(v.get_value(borrow=True)).any() for v in self.all_params]):
            raise RuntimeError('nan in some of the parameters')



queries_exp = EntityVectorLinkExp()
