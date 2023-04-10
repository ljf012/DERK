import random
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.DERK import Recommender
from utils.evaluate import test

from utils.helper import KGDataset
from utils.helper import collate_fn
from utils.helper import early_stopping

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def statistic_user_div_score(train_user_set, item_cate_set, n_cates):
    user_cate_div_item_ratio = {}
    for user in train_user_set:
        itemlist = train_user_set[user]
        len_item = len(itemlist)

        tmp_cate_set = set()
        for i in itemlist:
            c = item_cate_set[i]
            # for c_j in c:
            #     tmp_cate_set.add(c_j)
            tmp_cate_set.add(c)
                
    # Number of categories and number of interacted items
        if n_cates > len_item / 2:
            user_cate_div_item_ratio[user] = len(tmp_cate_set)/ (len_item+1e-6)
        else:
            user_cate_div_item_ratio[user] = len(tmp_cate_set)/ (n_cates+1e-6)

    return user_cate_div_item_ratio

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)

def _get_edges(graph):
    graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
    index = graph_tensor[:, :-1]  # [-1, 2]
    type = graph_tensor[:, -1]  # [-1, 1]
    return index.t().long(), type.long()


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    
def runner(rank, world_size, port):
    setup(rank, world_size, port)

    """fix the random seed"""
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()

    log = './training_log/' + args.dataset + '.log'
    sys.stdout = Logger(log, sys.stdout)

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, item_cate_set, entity_cate_set, cate_item_dict, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    n_cates = n_params['n_cates']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    user_div_score = statistic_user_div_score(user_dict['train_user_set'], item_cate_set, n_cates)

    train_data = KGDataset(train_cf, user_dict['train_user_set'], item_cate_set, cate_item_dict, user_div_score)
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, collate_fn=collate_fn)

    edge_index, edge_type = _get_edges(graph)
    adj_mat = _convert_sp_mat_to_sp_tensor(mean_mat_list[0])

    entity_cate_set = torch.tensor(list(entity_cate_set.values()))
    user_div_score = torch.tensor(list(user_div_score.values()))

    """define model"""
    model = Recommender(n_params, args, entity_cate_set.to(rank), edge_index.to(rank), edge_type.to(rank), adj_mat.to(rank)).to(rank)

    # DistributedDataParallel (DDP) for GPU multi-card parallel training
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        device_ids=[rank],
                                                        find_unused_parameters=True)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load(args.out_dir + 'model_' + args.dataset + '.ckpt')
        model.module.load_state_dict(ckpt)

        test_s_t = time()
        ret = test(rank, model.module, user_dict, n_params, item_cate_set, user_div_score)
        test_e_t = time()

        test_res = PrettyTable()
        test_res.field_names = ["tesing time", "recall", "ndcg", "precision", "hit_ratio",
        "category_coverage",
        "intralist_distance",
        "gini_index",
        "F-measure"]
        test_res.add_row(
            [test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio'], 
            ret['category_coverage'], ret['intralist_distance'], ret['gini_index'], 
            2*ret['recall']*ret['category_coverage'] / (ret['recall']+ret['category_coverage'])
            ]
        )
        print(test_res)
        return

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""

        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        
        train_sampler.set_epoch(epoch) #shuffle
        for batch in train_loader:
            batch_loss, _, _, batch_cor = model(batch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            cor_loss += batch_cor
            s += args.batch_size

        train_e_t = time()
        
        if rank == 0:
            if epoch % 20 == 0:
                """testing"""
                with torch.no_grad():
                    test_s_t = time()
                    ret = test(rank, model.module, user_dict, n_params, item_cate_set, user_div_score.to(rank))
                    test_e_t = time()

                    train_res = PrettyTable()
                    train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio",
                    "category_coverage", "intralist_distance", "gini_index",
                    "F-measure"]
                    train_res.add_row(
                        [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), 
                        ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio'], 
                        ret['category_coverage'], ret['intralist_distance'], ret['gini_index'], 
                        2*ret['recall']*ret['category_coverage'] / (ret['recall']+ret['category_coverage'])
                        ]
                    )
                    print(train_res)

                    # *********************************************************
                    # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                    cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                                stopping_step, expected_order='acc',
                                                                                flag_step=10)
                    if should_stop:
                        break

                """save weight"""
                if ret['recall'][0] == cur_best_pre_0 and args.save:
                    # torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')
                    torch.save(model.module.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

            else:
                # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
                print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@5:%.4f' % (epoch, cur_best_pre_0))

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    torch.multiprocessing.spawn(runner, args=(gpu_num, args.windows), nprocs=gpu_num, join=True)