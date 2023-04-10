import numpy as np
from sklearn.metrics import roc_auc_score
from .data_loader import find_cate


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def calculate_ndcg(rec_list, k, groundtruth):
    relevance_scores = np.zeros(len(rec_list[:k])).tolist()

    for i, item in enumerate(rec_list[:k]):
        if item in groundtruth:
            relevance_scores[i] = 1

    score = ndcg_at_k(relevance_scores, k, 1)
    return score
    
def recall_at_k(rank, k, ground_truth):
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


"""Diversity"""
def CC_at_k(ranklist, k, ground_truth, item_cate_set):
    """category Coverage
    category(r(i)) & category(all_user_item(i)) / category(all_user_item(i))
    """
    ranklist = ranklist[:k]
    catelist = find_cate(ranklist, item_cate_set)
    gt_cate = find_cate(ground_truth, item_cate_set)
    
    return len(set(catelist) & set(gt_cate)) / float(len(set(gt_cate)))


def ILD_at_k(ranklist, k, item_cate_set):
    """Intra-List Distance using Hamming distance

    """
    # k = len(ranklist)
    ranklist = ranklist[:k]
    tmp = 0
    catelist = find_cate(ranklist, item_cate_set)
    for i in range(len(catelist)):
        for j in range(i, len(catelist)):
            if catelist[i] != catelist[j]:
                # The category vector of two items calculates the cos distance, 
                # which is equivalent to the same category plus 1
                tmp += 1
    tmp = 2*tmp/(len(catelist)*(len(catelist)-1))
    return tmp
    

def gini_index(ranklist, k, item_cate_set):
    """Gini index"""
    cate = find_cate(ranklist[:k], item_cate_set)
    count = np.sort(np.unique(cate, return_counts=True)[1])
    n = len(count)
    cum_count = np.cumsum(count)

    return (n + 1 - 2 * np.sum(cum_count) / cum_count[-1]) / n