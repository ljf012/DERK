import torch
from torch.utils.data import Dataset
import numpy as np
import random

class KGDataset(Dataset):
    def __init__(self, entity_pairs, user_set, item_cate_set, cate_item_set, user_div_score):
        self.entity_pairs = entity_pairs
        self.user_set = user_set
        self.item_cate_set = item_cate_set
        self.cate_item_set = cate_item_set
        self.user_div_score = user_div_score

    def __getitem__(self, index):
        user = self.entity_pairs[index][0]
        item = self.entity_pairs[index][1]
        cate = self.item_cate_set[item]

        user_item = self.user_set[user]
        item_cate_set = self.item_cate_set
        cate_item_set = self.cate_item_set
        user_alpha = self.user_div_score[user]

        return user, item, cate, user_item, item_cate_set, cate_item_set, user_alpha

    def __len__(self):
        return len(self.entity_pairs)

def div_item_sample(user_item, item_cate_set, cate_item_set, user_alpha):

    clist = []
    cate_weight, cate_item = dict(), dict()

    ilist = user_item
    # ilist = list(ilist)
    if len(ilist) == 0:
        return
    # clist = item_cate_set[ilist]
    for i in ilist:
        clist.append(item_cate_set[i])

    # (Eq. 5)
    for item, cate in zip(ilist, clist):
        if cate not in cate_weight:
            cate_weight[cate] = 0
        if cate not in cate_item:
            cate_item[cate] = []
        cate_weight[cate] += 1
        cate_item[cate].append(item)
    max_weight = max(cate_weight.values())
    for cate, weight in cate_weight.items():
        cate_weight[cate] = max_weight / weight
    sum_weight = sum(cate_weight.values())
    for cate, weight in cate_weight.items():
        cate_weight[cate] = weight / sum_weight

    #reversed positive items
    samp_cate_list = np.random.choice(list(cate_weight.keys()), size=len(ilist), replace=True, p=list(cate_weight.values()))

    idx = random.randint(0, len(samp_cate_list)-1)
    cate = samp_cate_list[idx]

    if len(cate_item[cate]) < 2:
        i = np.random.choice(cate_item_set[cate])
    else:
        i = np.random.choice(cate_item[cate])

    if np.random.rand() >= user_alpha:
        i = ilist[idx]
    else:
        i = i
    reverse_item = int(i)

    # negative sampling
    if np.random.rand() < 0.8 or len(cate_item_set[cate]) < 5:
        neg_item = np.random.randint(1, len(item_cate_set))
        while neg_item in ilist:
            neg_item = np.random.randint(1, len(item_cate_set))
    else:
        neg_item = np.random.choice(cate_item_set[cate])
        while neg_item in ilist:
            neg_item = np.random.choice(cate_item_set[cate])
    reverse_neg = neg_item

    return reverse_item, reverse_neg

def negative_sampling(user_item, item_cate_set):
    while True:
        neg_item = np.random.randint(low=0, high=len(item_cate_set), size=1)[0]
        if neg_item not in user_item:
            break
    return neg_item

def collate_fn(batch_data):

    users, items, cates = [], [], []
    user_set, user_alpha_set = [], []
    neg_items, reverse_items, reverse_negs = [], [], []

    for user, item, cate, user_item, item_cate_set, cate_item_set, user_alpha in batch_data:
        users.append(user)
        items.append(item)
        cates.append(cate)
        user_set.append(user_item)
        user_alpha_set.append(user_alpha)

        neg_items.append(negative_sampling(user_item, item_cate_set))

        reverse_item, reverse_neg = div_item_sample(user_item,
                                                    item_cate_set,
                                                    cate_item_set, 
                                                    user_alpha)
        reverse_items.append(reverse_item)
        reverse_negs.append(reverse_neg)

    feed_dict = {}
    feed_dict['users'] = torch.LongTensor(users)
    feed_dict['pos_items'] = torch.LongTensor(items)
    feed_dict['neg_items'] = torch.LongTensor(neg_items)
    feed_dict['div_items'] = torch.LongTensor(reverse_items)
    feed_dict['div_neg_items'] = torch.LongTensor(reverse_negs)
    feed_dict['user_alpha'] = torch.FloatTensor(user_alpha_set)

    return feed_dict

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop