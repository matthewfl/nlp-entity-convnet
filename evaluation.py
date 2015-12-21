def evalCurrentState(queries, trainingData=True, numSamples=50000):
    all_measured = 0
    all_correct = 0
    all_trained = 0
    all_wrong = 0
    for qu in queries.values():
        for en in qu.values():
            if en['training'] != trainingData:
                continue
            if en['gold']:
                if all_trained > numSamples:
                    break
                all_measured += 1
                all_trained += len(en['vals'].values())
                sv = sorted(en['vals'].items(), key=lambda x: x[1])
                #m = max(en['vals'].values())
                if sv[-1][0] in en['gold']:
                    all_correct += 1
                else:
                    if len(set(en['gold']) & set(en['vals'].keys())) != 0:
#                         print sv[-1][0], en['gold']
                        all_wrong += 1
#                 for g in en['gold']:
#                     if en['vals'].get(en['gold'][0]) == m and m != 0:
#                         all_correct += 1
#                         break

    r = all_measured, all_wrong, float(all_correct) / all_measured
    print r
    return r


def evalCurrentStateRank(queries, trainingData=True, numSamples=50000):
    all_measured = 0
    all_correct_place = 0
    p_counts = dict((k,0) for k in range(0,10))
    all_trained = 0
    for qu in queries.values():
        for en in qu.values():
            if en['training'] != trainingData:
                continue
            if en['gold']:
                if all_trained > numSamples:
                    break
                svals = sorted(en['vals'].values(), key=lambda x: 0 if not isinstance(x, tuple) else -x[0])
                gv = en['vals'][en['gold']]
                if gv == 0:
                    continue
                all_measured += 1
                for i in xrange(len(svals)):
                    if svals[i] == gv:
                        if i < 10:
                            p_counts[i] += 1
                        all_correct_place += i + 1
                        break

    r = all_measured, float(all_correct_place) / all_measured, p_counts
    print r
    return r


def evalCurrentStateF1(queries, trainingData=True, numSamples=50000):
    correct = 0
    precDenom = 0
    recDenom = 0
    all_trained = 0
    all_measured = 0
    for qu in queries.values():
        if qu.values()[0]['training'] != trainingData:
            continue
        allGold = set()
        allChoosen = set()
        if all_trained > numSamples:
            break
        for en in qu.values():
            if en['gold']:  # we can eval this item
                all_measured += 1
                allGold.add(en['gold'])
                svals = sorted(en['vals'].values())
                picked = None
                for k,v in en['vals'].iteritems():
                    all_trained += 1
                    if v == svals[-1]:
                        picked = k
                allChoosen.add(picked)
                #if svals[0] == svals[1] and en['gold'] != picked:
                #    raise NotImplementedError()
#                 if en['gold'] == picked:
#                     correct += 1
#             if len(svals) > 5 and en['gold'] != picked:
#                 raise NotImplementedError()
        precDenom += len(allChoosen)
        recDenom += len(allGold)
        correct += len(allGold & allChoosen)
    correct = float(correct)
    prec = correct / precDenom
    rec = correct / recDenom
    f1 = 2 * prec * rec / (prec + rec)
    r = all_measured, 'Prec = {}/{} = {}, Rec = {}/{} = {}, F1 = {}'.format(
        correct, precDenom, prec,
        correct, recDenom, rec,
        f1)
    print r
    return r


from collections import defaultdict
def evalCurrentStateFahrni(queries, trainingData=True, numSamples=50000):
    def renderF1(corr, precDenom, recDenom):
        prec = float(corr) / precDenom
        rec = float(corr) / recDenom
        return 'Prec = {}/{} = {}, Rec = {}/{} = {}, F1 = {}'.format(
            corr, precDenom, prec,
            corr, recDenom, rec,
            2 * prec * rec / (prec + rec)
        )

    counter = defaultdict(lambda: 0)
    all_trained = 0
    for qu in queries.values():
        if qu.values()[0]['training'] != trainingData:
            continue
        for en in qu.values():
            if en['gold']:
                if all_trained > numSamples:
                    continue
                gold = en['gold']
                svals = sorted(en['vals'].items(), key=lambda x: x[1])
                picked = svals[-1][0]
                all_trained += len(svals)
                label = None
                if len(gold) == 1 and gold[0] == '-NIL-':
                    if picked == '-NIL-':
                        label = 'cNIL'
                    else:
                        label = 'wNIL_KB'
                elif picked in gold:
                    label = 'cKB'
                elif picked == '-NIL-':
                    label = 'wKB_NIL'
                else:
                    label = 'wKB_KB'
                counter[label] += 1

    rr = 'KB: {}'.format(renderF1(counter['cKB'], counter['cKB'] + counter['wKB_KB'] + counter['wNIL_KB'], counter['cKB'] + counter['wKB_KB'] + counter['wKB_NIL']))
    print rr
    if counter['cNIL']:
        rr2 = 'NIL: {}'.format(renderF1(counter['cNIL'], counter['cNIL'] + counter['wKB_NIL'], counter['cNIL'] + counter['wNIL_KB']))
        print rr2
        rr += '; '+rr2

    # print 'KB:', renderF1(counter['cKB'], counter['cKB'] + counter['wKB_KB'] + counter['wNIL_KB'], counter['cKB'] + counter['wKB_KB'] + counter['wKB_NIL'])
    # if counter['cNIL']:
    #     print 'NIL:', renderF1(counter['cNIL'], counter['cNIL'] + counter['wKB_NIL'], counter['cNIL'] + counter['wNIL_KB'])
    return counter, rr


def findWrongItems(queries, trainingData=True, numSamples=50):
    # theano overrides map if imported with *
    #from __builtin__ import map
    ret = []
#     surface = set()
    for qu in queries.values():
        for ek, en in qu.items():
            if en['training'] != trainingData:
                continue
#             for e in en:
            if en['gold']:
                if len(ret) > numSamples:
                    return ret
                if True:#ek not in surface:
                    sv = sorted(en['vals'].items(), key=lambda x: x[1])
                    if sv[-1][0] not in en['gold'] and len(set(en['gold']) & set(en['vals'].keys())) != 0:
#                     if 'Slayer (Buffy the Vampire Slayer)' in en['gold']:
                        # got this wrong
                        ret.append({
                            'gold': en['gold'],
                            'ordered': [(s[0], s[1][0], [(' '.join(map(str, a)), len(a)) for a in s[1][1]]) for s in sv][::-1],
                            'text': ek,
                            'training': en['training'],
                        })

#                     m = max(en['vals'].values())
#                     g = en['vals'].get(en['gold'][0], 0)
#                     if g != m and g != 0:
#                         ret[ek] = en
    return ret


def evalNumPossible(queries, qtype=(False,True)):
    total = 0
    possible = 0
    for qu in queries.values():
        for q in qu.values():
            if q['training'] in qtype:
                total += 1
                if len(set(q['gold']) & set(q['vals'].keys())) != 0:
                    possible += 1
    return float(possible) / total


def findItms(queries, key):
    ret = []
    from __builtin__ import map
    for qu in queries.values():
        for en in qu.values():
            if en['training'] == True:
                continue
            ad = False
            for k in en['vals'].keys():
                if key in k:
                    ad = True
            if ad:
                for k, v in en['vals'].iteritems():
                    ret.append((k, [(' '.join(map(str, a)), len(a)) for a in v[1]]))
#             return ret
    print len(ret)
    return ret
