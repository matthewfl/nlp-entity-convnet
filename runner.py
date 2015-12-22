import argparse
import json
import csv
import sys
import atexit
import os
import re
import cPickle as pickle

import h5py

from evaluation import *

from wordvecs import WordVectors

from baseWikipediaLinker import PreProcessedQueries

queries = None
featureNames = None
surface_counts = None
wordvectors = None

# result save files
csv_f = None
h5_f = None

# baseModel
baseModel = None

# the ml model
queries_exp = None

debug_log = []
results_log = []


def cleanWhitespaces():
    # this is just a clean up as a result of the query generator
    for qu in queries.values():
        for en in qu.values():
            if any([g.strip() != g for g in en['gold']]):
                # the gold never appear to have to be stripped
                raise RuntimeError()
            nv = {}
            for k, v in en['vals'].iteritems():
                # remove all items that contain extra whitespace
                # since these are never the gold items
                if k.strip() == k:
                    nv[k] = v
            en['vals'] = nv

def loadQueries(fname):
    global queries, featureNames
    with open(fname) as f:
        q = json.load(f)
        queries = q['queries']
        featureNames = q['featIndex']
    for q,v in queries.items():
        for v2 in v.values():
            gi = set(g.replace('_', ' ') for g in v2['gold']) | set(v2['gold'])
            v2['gold'] = list(gi)
    cleanWhitespaces()


def loadSurfaceCount(fname):
    global surface_counts
    with open(fname) as f:
        surface_counts = json.load(f)
    # try and make the surfaces items match what we are looking for
    surface_counts_re = re.compile('([\.,!\?])')
    for sk in surface_counts.keys():
        nsk = sk.replace('(', '-lrb-').replace(')', '-rrb-')
        nsk = surface_counts_re.sub(' \\1', nsk)
        if nsk != sk:
            surface_counts[nsk] = surface_counts[sk]

def loadWordVectors(wv_fname, redir_fname):
    global wordvectors
    wordvectors = WordVectors(
        fname=wv_fname, #"/data/matthew/enwiki-20141208-pages-articles-multistream-links7-output1.bin",
        redir_fname=redir_fname, #'/data/matthew/enwiki-20141208-pages-articles-multistream-redirect7.json',
        negvectors=False,
        sentence_length=200,
    )
    wordvectors.add_unknown_words = False


def argsp():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--queries', help='json file of the queries to run', required=True)
    aparser.add_argument('--surface_count', help='json file of link surface counts', required=True)
    aparser.add_argument('--wordvecs', help='the word vectors from word2vec', required=True)
    aparser.add_argument('--redirects', help='json of the redirects on wikipedia', required=True)
    aparser.add_argument('--wiki_dump', help='raw wiki dump file', required=True)
    aparser.add_argument('--batch_size', help='size of training batch', type=int, default=250)
    #aparser.add_argument('--dim_vec_compared', help='size of the vectors to compare for cosine-sim', type=int, default=150)
    aparser.add_argument('--num_iter', help='number of training iterations', type=int, default=10)
    aparser.add_argument('--raw_output', help='h5py file that represents raw information about this run', required=True)
    aparser.add_argument('--csv_output', help='csv results from this run', required=True)
    aparser.add_argument('--exp_model', help='the file to load for the experiment', default='exp_multi_conv_cosim')

    return aparser

def save_results():
    global debug_log, results_log, csv_f, h5_f

    if len(results_log) != 0:
        csv_f.writerow(['Results:'])
        csv_f.writerows(results_log)
        h5_f['results'] = results_log

    if len(debug_log) != 0:
        csv_f.writerow(['Log:'])
        csv_f.writerows(debug_log)
        h5_f['debug'] = pickle.dumps(debug_log)


def potentially_rename_file(fname):
    # make sure that we are saving this into a new file every time
    n = 1
    s = fname.split('.')
    s.insert(-1, '1')
    while os.path.isfile(fname):
        fname = '.'.join(s)
        n += 1
        s[-2] = str(n)
    return fname

def main():
    args = argsp().parse_args()

    global csv_f, h5_f

    # setup the save files
    csv_f = csv.writer(open(potentially_rename_file(args.csv_output), 'w'))
    csv_f.writerow(['Arguments:', ' '.join(sys.argv)])
    csv_f.writerow([])

    h5_f = h5py.File(potentially_rename_file(args.raw_output), 'w')
    h5_running_info = h5_f.create_group('meta_info')
    h5_running_info['arguments'] = sys.argv

    atexit.register(save_results)

    # load the queries
    loadQueries(args.queries)

    total_num_possible = evalNumPossible(queries)
    testing_num_possible = evalNumPossible(queries, (False,))
    h5_running_info['total_possible'] = total_num_possible
    h5_running_info['testing_possible'] = testing_num_possible
    csv_f.writerow(['Total queries possible', total_num_possible])
    csv_f.writerow(['Testing queries possible', total_num_possible])
    print 'Total queries possible: {}, Testing queries possible: {}'.format(total_num_possible, testing_num_possible)

    # load the word vectors and redirects
    loadWordVectors(args.wordvecs, args.redirects)
    print 'Number word vectors: {}'.format(len(wordvectors.vectors))

    # load the surface counts
    loadSurfaceCount(args.surface_count)

    # construct the base wikipedia information given the currently loaded queries, redirects, etc
    print 'Finding relevant pages from wikipedia'
    global baseModel
    baseModel = PreProcessedQueries(args.wiki_dump, wordvectors, queries, wordvectors.redirects, surface_counts)

    print 'Loading model'
    global queries_exp
    queries_exp = __import__(args.exp_model).queries_exp

    queries_exp.num_training_items = 50000000  # max number of items 50,000,000

    queries_exp.batch_size = args.batch_size

    # run the model
    results_log.append(('todo',))
    for i in xrange(args.num_iter):
        # run the training step
        print 'Training step', i
        res = ('Training step', i, queries_exp.compute_batch())
        debug_log.append(res)
        print res
        tstate = ('training state', evalCurrentState(queries, True, queries_exp.num_training_items))
        print tstate
        debug_log.append(tstate)
        # run the testing step
        tres = ('Testing step', queries_exp.compute_batch(False))
        print tres
        debug_log.append(tres)
        tstate = ('testing state', evalCurrentState(queries, False, queries_exp.num_training_items))
        print tstate
        debug_log.append(tstate)
        f1_res = evalCurrentStateFahrni(queries, False, queries_exp.num_training_items)
        debug_log.append([str(dict(f1_res[0])), f1_res[1]])







if __name__ == '__main__':
    main()
