# -*- coding:utf-8 -*-
# copyright: ykomashiro@gmail.com
import numpy as np
from math import log2


def uniquecounts(rows):
    results = {}
    for row in rows:
        r = row[len(row-1)]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def entropy(rows):
    """
    calculate the entropy of input data
        :param rows: the input data, some or whole data. 
    """
    results = uniquecounts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r])/len(rows)
        ent = ent-p*log2(p)
    return ent


class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def giniimpurity_2(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts.keys():
        p1 = float(counts[k1])/total
        imp += p1*(1-p1)
    return imp


def divideset(rows, column, value):
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function(row): return row[column] >= value
    else:
        split_function(row): return row[column] == value
    set1 = [row for row in rows if split_function(row)]
    se12 = [row for row in rows if not split_function(row)]
    return(set1, set2)


def buildtree(rows, scoref=entropy):
    if len(rows) == 0:
        return decisionnode()
    current_score = scoref(rows)
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0])-1
    for col in range(0, column_count):
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)
            p = float(len(set1))/len(rows)
            gain = current_score-p*scoref(set1)-(1-p)*scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


def printtree(tree, indent=''):
    if tree.results != None:
        print(str(tree.results))
    else:
        print(str(tree.col)+':'str(tree.value)+'? ')
        print(indent+"T->")
        printtree(tree.tb, indent+" ")
        print(indent+"F->")
        printtree(tree.fb, indent+" ")
def classify()