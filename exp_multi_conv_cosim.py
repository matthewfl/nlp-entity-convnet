from theano import *
from lasagne.layers import InputLayer, get_output
import lasagne
import lasagne.layers
import theano.tensor as T
import theano
import numpy as np
from helpers import SimpleMaxingLayer, SimpleAverageLayer
from wordvecs import WordVectors, EmbeddingLayer, WordTokenizer
#import json
import re
import random

theano.config.floatX = 'float32'
#theano.config.linker = 'cvm_nogc'
theano.config.openmp = True
theano.config.openmp_elemwise_minsize = 20000

from runner import baseModel


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

        #self.x_indicator_features = T.matrix('x_indicator_features', dtype='int8')

        self.x_denotaiton_features = T.matrix('x_denotation_ind_feats', dtype='int8')  # the joint denotation query features
        self.x_query_featurs = T.matrix('x_query_ind_feats', dtype='int8')  # the query features
        self.x_query_link_id = T.ivector('x_match_query')  # the query that a denotation links to
        self.x_denotation_ranges = T.imatrix('x_denotation_ranges')  # the range of joint denotations to sum over

        self.x_target_link_id = T.ivector('x_match_target')  # the target document that maches with a given denotation

        #self.y_score = T.vector('y')
        #self.y_answer = T.ivector('y_ans')  # (Not used) contains the location of the gold answer so we can compute the loss
        self.y_isgold = T.vector('y_gold', dtype='int8')  # is 1 if the gold item, 0 otherwise
        self.y_grouping = T.imatrix('y_grouping')  # matrix containing [start_idx, end_idx, gold_idx]
        self.y_boosted = T.vector('y_boosted')  # only used if boosting enabled, vector of how much to boost items

        self.embedding_W = theano.shared(self.wordvecs.get_numpy_matrix().astype(theano.config.floatX),name='embedding_W')
        self.embedding_W_docs = theano.shared(self.documentvecs.get_numpy_matrix().astype(theano.config.floatX),name='embedding_W_docs')

#         def convSoftMax(t):
#             shape = t.shape  # (document_sample, num_filters, output_rows, output_cols)
#             new_shape = (shape[0] * shape[2] * shape[3], shape[1])
#             return T.nnet.softmax(t.reshape(new_shape)).reshape(shape)

#             return lasagne.nonlinearities.leaky_rectify(t)

        def augRectify(x):
            # if x is zero, then the gradient failes due to computation: x / |x|
            #return lasagne.nonlinearities.leaky_rectify(x + .001) #- .001
            return T.maximum(x, -.01 * x)

        simpleConvNonLin = augRectify # lasagne.nonlinearities.rectify# leaky_rectify

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
            filter_size=(2, self.wordvecs.vector_size),
            name='document_simple_conv',
            nonlinearity=simpleConvNonLin,#lasagne.nonlinearities.leaky_rectify,
        )

        self.document_simple_sum_l = lasagne.layers.Pool2DLayer(
            #lasagne.layers.reshape(self.document_embedding_l, ([0],[3],[2],1)),
            self.document_simple_conv1_l,
            name='document_simple_pool',
            pool_size=(self.document_length - 2, 1),
            mode='sum',
        )

#         self.document_conv1_l = lasagne.layers.Conv2DLayer(
#             self.document_embedding_l,
#             num_filters=30,  # was 75, 100, 500
#             filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
#             name='document_conv1',
#             nonlinearity=self.main_nl,# lasagne.nonlinearities.softmax,  # was leaky_rectify
#         )

#         self.document_max_l = lasagne.layers.Pool2DLayer(
#             self.document_conv1_l,
#             name='document_pool1',
#             pool_size=(self.document_length - self.num_words_to_use_conv, 1),
#             mode='max',  # was sum
#         )

#         document_output_length = 25 # was 100, 200

#         self.document_dens1 = lasagne.layers.DenseLayer(
#             self.document_max_l,
#             num_units=document_output_length,
#             name='doucment_dens1',
#             nonlinearity=lasagne.nonlinearities.leaky_rectify,
#         )

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
            num_filters=self.dim_compared_vec,  # was 300
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            name='surface_cxt_conv1',
            nonlinearity=simpleConvNonLin,# self.main_nl,
        )

        self.surface_context_pool1_l = lasagne.layers.Pool2DLayer(
            self.surface_context_conv1_l,
            name='surface_cxt_pool1',
            pool_size=(self.sentence_length - self.num_words_to_use_conv, 1),
            mode='max', #sum',
        )

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
            num_filters=self.dim_compared_vec,  # was 300
            filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
            name='surface_conv1',
            nonlinearity=simpleConvNonLin, #self.main_nl,
        )

        self.surface_pool1_l = lasagne.layers.Pool2DLayer(
            self.surface_conv1_l,
            name='surface_pool1',
            pool_size=(self.sentence_length_short - self.num_words_to_use_conv, 1),
            mode='sum',
        )

        self.surface_words_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.surface_pool1_l, ([0], -1))
        )

        self.all_params += lasagne.layers.get_all_params(self.surface_pool1_l)

        ##################################################
        ## merge the documents with the surface info


        ###################################################
        ## dealing with the target side

        matched_surface_reshaped = self.x_matches_surface.reshape(
            (self.x_matches_surface.shape[0], 1, 1, 1)).astype(theano.config.floatX)

        self.target_input_l = lasagne.layers.InputLayer(
            (None,),
            input_var=self.x_target_input
        )

        #################################
        ## target indicators features

        self.target_matched_surface_input_l = lasagne.layers.InputLayer(
            (None,1,1,1),
            input_var=matched_surface_reshaped,
        )

        self.target_matched_counts_input_l = lasagne.layers.InputLayer(
            (None,5),
            input_var=self.x_matches_counts.astype(theano.config.floatX),
        )

#         # embedding of the target documents
#         self.target_embedding_l = EmbeddingLayer(
#             lasagne.layers.reshape(self.target_input_l, ([0], 1)),
#             W=self.embedding_W_docs,
#             add_word_params=False,
#         )

#         self.target_embedding_dens_l = lasagne.layers.DenseLayer(
#             lasagne.layers.reshape(self.target_embedding_l, ([0], -1)),
#             name='target_embedding_dens',
#             num_units=self.dim_compared_vec,
#             nonlinearity=augRectify,# lasagne.nonlinearities.leaky_rectify,  # should be compariable to the simpleConvNonLin
#         )

#         self.target_embedding_out = lasagne.layers.get_output(
#             lasagne.layers.reshape(self.target_embedding_dens_l, ([0],-1)),
#         )

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
            num_filters=self.dim_compared_vec,  # was 75, 150, 350
            nonlinearity=simpleConvNonLin,# self.main_nl,# lasagne.nonlinearities.leaky_rectify,
        )

        self.target_words_pool1_l = lasagne.layers.Pool2DLayer(
            self.target_words_conv1_l,
            name='target_wrds_pool1',
            pool_size=(self.sentence_length_short - self.num_words_to_use_conv, 1),
            mode='sum',  # was sum
        )

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
            filter_size=(2, self.wordvecs.vector_size),
            num_filters=self.dim_compared_vec,
            nonlinearity=simpleConvNonLin,# self.main_nl,# lasagne.nonlinearities.leaky_rectify,
        )

        self.target_body_simple_sum_l = lasagne.layers.Pool2DLayer(
            #lasagne.layers.reshape(self.target_body_words_embedding_l, ([0],[3],[2],1)),
            self.target_body_simple_conv1_l,
            name='target_body_simple_sum',
            pool_size=(self.sentence_length - 2, 1),
            mode='sum',
        )

#         self.target_body_words_conv1_l = lasagne.layers.Conv2DLayer(
#             self.target_body_words_embedding_l,
#             name='target_body_wrds_conv1',
#             filter_size=(self.num_words_to_use_conv, self.wordvecs.vector_size),
#             num_filters=150,
#             nonlinearity=lasagne.nonlinearities.leaky_rectify,
#         )

#         self.target_body_words_pool1_l = lasagne.layers.Pool2DLayer(
#             self.target_body_words_conv1_l,
#             name='target_body_wrds_pool1',
#             pool_size=(self.sentence_length - self.num_words_to_use_conv, 1),
#             mode='max',
#         )

#         self.target_merge_l = lasagne.layers.ConcatLayer(
#             [lasagne.layers.reshape(self.target_words_pool1_l, ([0], [1])),
#              lasagne.layers.reshape(self.target_body_words_pool1_l, ([0], [1])),
#             # lasagne.layers.reshape(self.target_embedding_l, ([0], [3]))
#             ]
#         )

        self.target_out = lasagne.layers.get_output(
            lasagne.layers.reshape(self.target_body_simple_sum_l, ([0],-1)))

        self.all_params += lasagne.layers.get_all_params(self.target_body_simple_sum_l)

        #########################################################
        ## compute the cosine distance between the two layers

        # source body
        self.source_aligned_l = self.document_output[self.x_document_id,:][self.x_link_id,:] #self.source_out[self.x_link_id, :]
        # source context
        self.source_context_aligned_l = self.surface_output[self.x_link_id,:]
        # source surface words
        self.source_surface_words_aligned_l = self.surface_words_output[self.x_link_id,:]

        # this uses scan internally, which means that it comes back into python code to run the loop.....fml
#         self.dotted_vectors =  T.batched_dot(self.target_out, self.source_aligned_l)
        # diag also does not support a C version.........
        #self.dotted_vectors = T.dot(self.target_out, self.source_aligned_l.T).diagonal()

        def augNorm(v):
            return T.basic.pow(T.basic.pow(T.basic.abs_(v), 2).sum(axis=1) + .001, .5)

#         self.res_l = self.dotted_vectors / (augNorm(self.target_out) * augNorm(self.source_aligned_l) + .001)

#         self.res_cap = T.clip((T.tanh(self.res_l) + 1) / 2, .001, .999)

        def cosinsim(a, b):
            dotted = T.batched_dot(a, b)
            return dotted / (augNorm(a) * augNorm(b))

        ##############################################
        ## tensor product stuff

#         def tensorP(a,b):
#             res, _ = theano.scan(
#                 fn=lambda x_vec, y_vec, x_norm, y_norm: T.concatenate(
#                     [x_vec, y_vec,
#                      T.outer(x_vec / x_norm, y_vec / y_norm).flatten()]
#                 ),
#                 outputs_info=None,
#                 sequences=[a,b, augNorm(a), augNorm(b)],
#                 non_sequences=None
#             )
#             return res


        def comparedVLayers(a, b):
            dv = cosinsim(a, b)
            #dv = (a[:,0] + .001) * (b[:,0] + .001)
            return lasagne.layers.InputLayer(
                (None,1),
                input_var=dv.reshape((dv.shape[0], 1))
            )

#         def comparedVLayers2(a,b):
#             dv = theano.gradient.disconnected_grad(tensorP(a,b))  # just see how well this performs without learning this
#             return lasagne.layers.InputLayer(
#                 (None,self.dim_compared_vec ** 2 + self.dim_compared_vec*2 ),
#                 input_var=dv
#             )




        ######################################################
        ## indicator feature input

#         self.indicator_feat_l = lasagne.layers.InputLayer(
#             (None, self.num_indicator_features),
#             input_var=self.x_indicator_features.astype(theano.config.floatX),
#         )

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

#         self.cosine_weighted.W.get_value(borrow=True)[:] += 1

        self.cosine_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.cosine_weighted, (-1,)))

        self.all_params += lasagne.layers.get_all_params(self.cosine_weighted)

        self.query_feat_l = lasagne.layers.InputLayer(
            (None,self.num_indicator_features),
            input_var=self.x_query_featurs,
        )

        rank_feats = [f[0] for f in enumerate(featuresNames) if f[1].startswith('Rank=')]

        self.denotation_join_feat_l = lasagne.layers.InputLayer(
            (None,self.num_indicator_features),
            input_var=self.x_denotaiton_features,#[:, rank_feats],
        )

        self.query_layer_l = lasagne.layers.DenseLayer(
            self.query_feat_l,
            name='query_lin',
            num_units=1,
            nonlinearity=lasagne.nonlinearities.linear,
        )

#         self.query_layer_l.b.get_value(borrow=True)[:] += 1

        self.query_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.query_layer_l, (-1,))
        )

#         self.all_params = []

#         self.all_params += lasagne.layers.get_all_params(self.query_layer_l)

        self.aligned_queries = self.query_output[self.x_query_link_id]

        self.aligned_cosine = self.cosine_output[self.x_target_link_id]

        self.denotation_layer_l = lasagne.layers.DenseLayer(
            self.denotation_join_feat_l,
            name='denotation_lin',
            num_units=1,
            nonlinearity=lasagne.nonlinearities.linear,
            W=self.query_layer_l.W,
        )

#         self.denotation_layer_l.b.get_value(borrow=True)[:] += 1

        self.denotation_output = lasagne.layers.get_output(
            lasagne.layers.reshape(self.denotation_layer_l, (-1,)))

        self.all_params += lasagne.layers.get_all_params(self.denotation_layer_l)

        ###########################
        ## multiply the two parts of the join scores

        #  self.aligned_cosine
        self.unmerged_scores =  (
            ( #(self.aligned_queries) +
            (self.denotation_output))
            + self.aligned_cosine
            # *
#             (1 - self.x_denotaiton_features[:, self.impossible_query])
        )

        # this should be the fastest way to compute this
        # however it is creating matrices too large and isn't able to allocate enough memory or something

#         def mergingSelector(indx, outputs):
#             return T.set_subtensor(outputs[indx[0], T.arange(indx[1],indx[2])], 1)

#         merging_seq = T.concatenate([
#                 T.arange(self.x_denotation_ranges.shape[0]).reshape((self.x_denotation_ranges.shape[0], 1)),
#                 self.x_denotation_ranges,
#         ], axis=1)

#         self.merging_matrix, _ = theano.scan(
#             mergingSelector,
#             outputs_info=T.zeros((self.x_denotation_ranges.shape[0], self.denotation_output.shape[0])),
#             sequences=merging_seq,
#         )

#         self.merged_scores = T.dot(self.merging_matrix, self.unmerged_scores)


        def sloppyMathLogSum(vals):
            m = vals.max()
            return T.log(T.exp(vals - m).sum()) + m

        def mergingSum(indx, unmerged):
            return sloppyMathLogSum(unmerged[T.arange(indx[0], indx[1])]) #.mean()

        self.merged_scores, _ = theano.scan(
            mergingSum,
            sequences=[self.x_denotation_ranges],
            non_sequences=[self.unmerged_scores]
        )

#         self.merged_scores = self.cosine_output

        # prevents the softmax from blowing up
#         self.merged_rescaled = 10 * self.merged_scores / theano.gradient.disconnected_grad(abs(self.merged_scores).max())


        #############################
        ## Linear features combined
        #############################


#         self.linear_features_combined = lasagne.layers.concat(
#             [

#                comparedVLayers(self.target_out, self.source_aligned_l),
#                comparedVLayers(self.target_out, self.source_context_aligned_l),
#                comparedVLayers(self.target_out, self.source_surface_words_aligned_l),

#                comparedVLayers(self.target_title_out, self.source_aligned_l),
#                comparedVLayers(self.target_title_out, self.source_context_aligned_l),
#                comparedVLayers(self.target_title_out, self.source_surface_words_aligned_l),


# #                 comparedVLayers(self.target_embedding_out, self.source_aligned_l),
# #                 comparedVLayers(self.target_embedding_out, self.source_context_aligned_l),
# #                 comparedVLayers(self.target_embedding_out, self.source_surface_words_aligned_l),

#             lasagne.layers.reshape(self.target_matched_surface_input_l, ([0],1)),
#             self.target_matched_counts_input_l,
#              self.indicator_feat_l
#             ],
#             axis=1
#         )

#         self.linear_features_dens_l = lasagne.layers.DenseLayer(
#             self.linear_features_combined,
#             nonlinearity=lasagne.nonlinearities.linear,  # use tanh so that too large of values don't cause the softmax issues
#             num_units=1,
#             name='linear_final_l',
#             W=lasagne.init.Normal(mean=0.0),
#         )

#         self.linear_features_dens_l.W.get_value(borrow=True)[0:9] += 1.0  # set the word vecs positive
# #         lin_feat_W = self.linear_features_dens_l.W.get_value(borrow=True)
# #         lin_feat_add_one = np.eye(self.dim_compared_vec).reshape(self.dim_compared_vec**2, 1)
# #         lin_feat_W[self.dim_compared_vec*2                           :self.dim_compared_vec*2+  self.dim_compared_vec**2] += lin_feat_add_one
# #         lin_feat_W[self.dim_compared_vec*4+  self.dim_compared_vec**2:self.dim_compared_vec*4+2*self.dim_compared_vec**2] += lin_feat_add_one
# #         lin_feat_W[self.dim_compared_vec*6+2*self.dim_compared_vec**2:self.dim_compared_vec*6+3*self.dim_compared_vec**2] += lin_feat_add_one

#         self.linear_output = lasagne.layers.get_output(
#             lasagne.layers.reshape(self.linear_features_dens_l, ([0],))
#         )

#         # rescaled the output so we don't crash the softmax layer with a value that is too large
#         # just a hack
#         self.linear_output_rescaled = 10 * self.linear_output / theano.gradient.disconnected_grad(abs(self.linear_output).max())

        ########################################
        ## true output values
        ########################################

        self.unscaled_output = self.merged_scores

        def scaleRes(indx, outputs, res):
            #print 'scaleRes'
            ran = T.arange(indx[0], indx[1])
            s = sloppyMathLogSum(res[ran])
            return T.set_subtensor(outputs[ran], res[ran] - s)

        self.scaled_scores, _ = theano.scan(
            scaleRes,
            sequences=[self.y_grouping],
            non_sequences=[self.unscaled_output],
            outputs_info=T.zeros((self.unscaled_output.shape[0],))
        )

#         self.scaled_scores = scaled_output_grp.flatten()


#         self.all_params = []
        # rescale the scores to prevent softmax from blowing up
        #self.true_output = T.zeros((self.unscaled_output.shape[0],))
        self.true_output =  self.scaled_scores[-1]
        #self.merged_scores #10 * self.unscaled_output / theano.gradient.disconnected_grad(abs(self.unscaled_output).max())#self.cosine_output # self.merged_rescaled #self.linear_output_rescaled



#         self.res_l = self.dotted_vectors / ((self.target_out.norm(1, axis=1) + .001) *
#                                             (self.source_aligned_l.norm(1, axis=1) + .001))


        #self.golds = self.res_cap[self.y_answer]

#         def maxOverRange(indx):
#             #return T.max(self.res_cap[T.arange(indx[0],indx[1])]) - self.res_cap[indx[2]]
#             #return -( self.res_l[indx[2]] - T.log(T.exp(self.res_l[T.arange(indx[0],indx[1])]).sum()) )
#             return -( self.res_l[indx[2]] - self.res_l[indx[0]])

#         # build a tensor to make a matrix with one set on each dimention
#         self.grouped, grouped_update = theano.scan(maxOverRange, sequences=self.y_grouping)

###########
#         def setSubSelector(indx, outputs):
#             return T.set_subtensor(outputs[T.arange(indx[0], indx[1]), indx[3]], 1)

#         num_target_samples = self.true_output.shape[0]

#         select_seq = T.concatenate([
#             self.y_grouping,
#             T.arange(self.y_grouping.shape[0]).reshape((self.y_grouping.shape[0], 1))
#         ], axis=1)

#         self.selecting_matrix, _ = theano.scan(
#             setSubSelector,
#             outputs_info=T.zeros((num_target_samples, self.y_grouping.shape[0])), #num_target_samples)),
#             #n_steps=self.y_grouping.shape[0]
#             sequences=select_seq,
#         )

#########


#         self.groupped_elems = T.dot(self.selecting_matrix[-1].T,
#                                    T.exp(self.true_output))
#         self.groupped_res = T.log(self.groupped_elems)

        def lossSum(indx, res):
            return sloppyMathLogSum(res[T.arange(indx[0], indx[1])])

        self.groupped_res, _ = theano.scan(
            lossSum,
            sequences=[self.y_grouping],
            non_sequences=[self.true_output],
        )

        def selectGolds(indx, res, golds):
            r = T.arange(indx[0], indx[1])
            # the gold value should simply come from the input
            # so there is no good reason to have to disconnect the graident here
            gs = theano.gradient.disconnected_grad(golds[r])
            vals = gs * res[r] + (1 - gs) * -1000000  # approx 0
            return sloppyMathLogSum(vals)

        self.gold_res, _ = theano.scan(
            selectGolds,
            sequences=[self.y_grouping],
            non_sequences=[self.true_output, self.y_isgold],
        )

        self.loss_vec = self.groupped_res - self.gold_res #self.true_output[self.y_grouping[:,2]]

        if self.enable_boosting:
            self.loss_scalar = T.dot(self.y_boosted, self.loss_vec)
        else:
            self.loss_scalar = self.loss_vec.sum()

#         self.all_params = (
#             lasagne.layers.get_all_params(self.document_simple_sum_l) +
#             lasagne.layers.get_all_params(self.surface_context_pool1_l) +
#             lasagne.layers.get_all_params(self.surface_pool1_l) +
#             lasagne.layers.get_all_params(self.target_body_simple_sum_l) +
#             lasagne.layers.get_all_params(self.target_words_pool1_l) +
#             lasagne.layers.get_all_params(self.linear_features_dens_l)
#         )

#         self.regularization = (self.linear_features_dens_l.W ** 2).sum() / 400

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

        self.updates = lasagne.updates.adadelta(
            self.loss_scalar / self.loss_vec.shape[0] , #+ self.regularization,
            self.all_params)

        self.func_inputs = [
            self.x_document_input,
            self.x_surface_text_input, self.x_surface_context_input, self.x_document_id,
            self.x_target_input, self.x_matches_surface, self.x_matches_counts, self.x_link_id,
            self.x_target_words, self.x_target_document_words, #self.x_indicator_features,
            self.x_denotaiton_features, self.x_query_featurs, self.x_query_link_id, self.x_denotation_ranges,
            self.x_target_link_id,
            #self.y_answer,
            self.y_grouping, self.y_boosted,
            self.y_isgold,
        ]

        self.func_outputs = [
            self.true_output,
            self.loss_vec.sum(),
            self.loss_scalar,
            self.loss_vec,
            #self.res_l,
        ]

        ################################################################3
        ## TODO: need to return the actual output layer instead of the res_cap, since that is something else now

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
        self.current_boosted_groups = []

        self.current_queries = []
        self.current_denotations_feats_indicators = []
        self.current_denotations_related_query = []
        self.current_denotations_range = []
        self.current_denotation_targets_linked = []

        self.failed_match = []

    def compute_batch(self, isTraining=True, useTrainingFunc=True):
        if isTraining and useTrainingFunc:
            func = self.train_func
        else:
            func = self.test_func
        self.reset_accums()
        self.total_links = 0
        self.total_loss = 0.0
        self.total_boosted_loss = 0.0

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
                    self.current_boosted_groups.append(targets['boosted'])

                    self.current_queries += queries_feats_indicators

                    self.current_denotations_feats_indicators += denotations_joint_indicators
                    self.current_denotations_related_query += denotations_linked_query
                    self.current_denotations_range += denotations_range

                    self.current_denotation_targets_linked += denotation_target_linked

                #self.current_target_goal.append(isGold)
                self.learning_targets += target_learings
            if len(self.current_target_id) > self.batch_size:
#                 return
                self.run_batch(func)
                if self.total_links > self.num_training_items:
                    return self.total_loss / self.total_links, self.total_boosted_loss / self.total_links

        if len(self.current_target_id) > 0:
            self.run_batch(func)

        return self.total_loss / self.total_links, self.total_boosted_loss / self.total_links

    def run_batch(self, func):
        res_vec, loss_sum, loss_boosted, loss_vec, = func(
            self.current_documents,
            self.current_surface_link, self.current_surface_context, self.current_link_id,
            self.current_target_input, self.current_target_matches_surface, self.current_surface_target_counts, self.current_target_id,
            self.current_target_words, self.current_target_body_words, #self.current_feat_indicators,
            self.current_denotations_feats_indicators, self.current_queries, self.current_denotations_related_query, self.current_denotations_range,
            self.current_denotation_targets_linked,
            #self.current_target_goal,
            self.current_learning_groups, self.current_boosted_groups,
            self.current_target_is_gold,
        )
        self.check_params()
        self.total_links += len(self.current_target_id)
        self.total_loss += loss_sum
        self.total_boosted_loss += loss_boosted
        learned_groups = []  # right...dict not hashable....
        for i in xrange(len(res_vec)):
            # save the results from this pass
            l = self.learning_targets[i]
            if l[1] in l[0]['vals']:
                l[0]['vals'][ l[1] ][0] = float(res_vec[i]), 0#float(nn_outs[i])
            if l[0] not in learned_groups:
                learned_groups.append(l[0])
#         for group in learned_groups:
#             if group['gold']:
#                 correct = max(group['vals']) == group['vals'].get(group['gold'])
#                 group['boosted'] *= .4 if correct else 2.0
#                 if self.enable_cap_boosting:
#                     if group['boosted'] > 10:
#                         group['boosted'] = 10.0
#                     elif group['boosted'] < 0.1:
#                         group['boosted'] = 0.1
        self.reset_accums()

    def check_params(self):
        if any([np.isnan(v.get_value(borrow=True)).any() for v in self.all_params]):
            raise RuntimeError('nan in some of the parameters')



queries_exp = EntityVectorLinkExp()
