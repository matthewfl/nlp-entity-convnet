import argparse
import json
import csv
import sys

from evaluation import *
from runner import potentially_rename_file

base_queries = None
base_featureNames = None
queries = None
featureNames = None

def argsp():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--base_queries', help='the queries to compare against', required=True)
    aparser.add_argument('--queries', help='the queries for the comparison', required=True)
    aparser.add_argument('--csv_output', help='csv file to save results')
    aparser.add_argument('--use_training', help='use training queries instead of test', action='store_true', dest='use_training')
    aparser.set_defaults(use_training=False)

    return aparser

def main():
    args = argsp().parse_args()

    global base_queries, queries, base_featureNames, featureNames
    with open(args.base_queries) as f:
         j = json.load(f)
         base_queries = j['queries']
         base_featureNames = j['featIndex']

    with open(args.queries) as f:
        j = json.load(f)
        queries = j['queries']
        featureNames = j['featIndex']

    csv_f = None
    if args.csv_output:
        csv_f = csv.writer(open(potentially_rename_file(args.csv_output), 'w'))
        csv_f.writerow(['Arguments:', ' '.join(sys.argv)])
        csv_f.writerow([])

        csv_f.writerow(['gold', 'before', 'context', 'document', 'indicators gold', 'indicators wrong', 'indicators added', 'indicators removed'])

    print 'base score:'
    evalCurrentStateFahrni(base_queries, args.use_training, 50000000)[1]
    print 'comp score:'
    evalCurrentStateFahrni(queries, args.use_training, 50000000)[1]
    print ''

    for qk, qu in queries.iteritems():
        if qu.values()[0]['training'] != args.use_training:
            # ignore the items we don't want
            continue
        qu2 = base_queries[qk]
        for ek, en in qu.iteritems():
            en2 = qu2[ek]
            sv = sorted(en['vals'].items(), key=lambda x: x[1])
            sv2 = sorted(en2['vals'].items(), key=lambda x: x[1])

            c = not (sv[-1][0] not in en['gold'] and len(set(en['gold']) & set(en['vals'].keys())) != 0)
            c2 = not (sv2[-1][0] not in en2['gold'] and len(set(en2['gold']) & set(en2['vals'].keys())) != 0)

            if c and not c2:
                # got correct now but not before
                feats_c = set()
                feats_w = set()
                for f in sv[-1][1][1]:
                    feats_c.update(f)
                for f in sv2[-1][1][1]:
                    feats_w.update(f)
                fn_c = ' '.join([featureNames[f] for f in feats_c])
                fn_w = ' '.join([featureNames[f] for f in feats_w])
                fn_a = ' '.join([featureNames[f] for f in (feats_c - feats_w)])
                fn_b = ' '.join([featureNames[f] for f in (feats_w - feats_c)])
                try:
                    print 'gold:', en['gold']
                    print 'before:', sv2[-1][0]
                    print 'now:', sv[-1][0]
                    print 'context:', ek
                    print 'indicators gold   :', fn_c
                    print 'indicators wrong  :', fn_w
                    print 'indiactors added  :', fn_a
                    print 'indicators removed:', fn_b
                    print 'choices:', en['vals'].keys()
                    if csv_f:
                        csv_f.writerow([en['gold'], sv2[-1][0], ek, qk, fn_c, fn_w, fn_a, fn_b])
                except UnicodeEncodeError as e:
                    pass
                print ''



if __name__ == '__main__':
    main()
