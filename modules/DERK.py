import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

def _edge_sampling(edge_index, edge_type, rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]

def _sparse_dropout(x, rate=0.5):
    noise_shape = x._nnz()

    random_tensor = rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    return out * (1. / (1 - rate))

class Cul_cor(nn.Module):
    def __init__(self, ind, temperature, n_factors):
        super(Cul_cor, self).__init__()
        self.ind = ind
        self.temperature = temperature
        self.n_factors = n_factors

    def CosineSimilarity(self, tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
    def DistanceCorrelation(self, tensor_1, tensor_2):
        # tensor_1, tensor_2: [channel]
        # ref: https://en.wikipedia.org/wiki/Distance_correlation
        channel = tensor_1.shape[0]
        zeros = torch.zeros(channel, channel).to(tensor_1.device)
        zero = torch.zeros(1).to(tensor_1.device)
        tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
        """cul distance matrix"""
        a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
        tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
        a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
        """cul distance correlation"""
        A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
        dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
        dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
        dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
        return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
    def MutualInformation(self, disen_weight_att):
        # disen_T: [num_factor, dimension]
        disen_T = disen_weight_att.t()

        # normalized_disen_T: [num_factor, dimension]
        normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

        pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
        ttl_scores = torch.sum(torch.mm(disen_T, disen_weight_att), dim=1)

        pos_scores = torch.exp(pos_scores / self.temperature)
        ttl_scores = torch.exp(ttl_scores / self.temperature)

        mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
        return mi_score

    def forward(self, disen_weight_att):
        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return self.MutualInformation(disen_weight_att)
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += self.DistanceCorrelation(disen_weight_att[i], disen_weight_att[j])
                    else:
                        cor += self.CosineSimilarity(disen_weight_att[i], disen_weight_att[j])
        return cor

class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.mlp(x)

class Aggregator(nn.Module):
    def __init__(self, n_users, n_cates, n_relations, n_factors, emb_size, is_div):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors
        self.is_div = is_div
        self.user_att = _MultiLayerPercep(n_factors, n_factors)
        if not is_div:
            self.weight_att = _MultiLayerPercep(n_relations - 1, n_relations - 1)
        else:
            self.weight_att = _MultiLayerPercep(n_cates + n_relations - 1, n_cates + n_relations - 1)
        self.w1 = nn.Linear(emb_size, emb_size)
        self.w2 = nn.Linear(emb_size, emb_size)


    def forward(self, entity_emb, user_emb, latent_emb,
                edge_index, edge_type, interact_mat,
                weight, entity_cate_set):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """KG aggregate"""
        # (Eq. 6)
        head, tail = edge_index
        if not self.is_div:
            edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
            neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
            # e_i
            entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        else:
            edge_cate_emb = weight[entity_cate_set[tail]-1]   # exclude interact, remap [1, n_cates) to [0, n_relations-1)
            neigh_cate_emb = entity_emb[tail] * edge_cate_emb  # [-1, channel]
            entity_agg = scatter_mean(src=neigh_cate_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]

        """cul user->latent interest attention"""
        # beta(u,a) (Eq. 8)
        score_ = torch.mm(self.w1(user_emb), self.w1(latent_emb).t())
        score = nn.Softmax(dim=1)(self.user_att(score_)).unsqueeze(-1)  # [n_users, n_factors, 1]

        # e_a (Eq. 1) (Eq. 2)
        score_rp = torch.mm(self.w2(latent_emb), self.w2(weight).t())
        latent_emb = torch.mm(nn.Softmax(dim=-1)(self.weight_att(score_rp)), weight)
        disen_weight = latent_emb.expand(n_users, n_factors, channel)

        # (Eq. 7)
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg # [n_users, channel]

        return entity_agg, user_agg, latent_emb


class GraphConv(nn.Module):
    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, n_cates, interact_mat, emb_size,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.extends = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        weight_d = initializer(torch.empty(n_cates + n_relations - 1, channel))  # not include interact
        self.weight_d = nn.Parameter(weight_d)  # [n_cates - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_cates=n_cates, n_relations=n_relations, n_factors=n_factors, emb_size=emb_size, is_div=False))
            self.extends.append(Aggregator(n_users=n_users, n_cates=n_cates, n_relations=n_relations, n_factors=n_factors, emb_size=emb_size, is_div=True))

        self._cor = Cul_cor(ind=ind, temperature=self.temperature, n_factors=n_factors)

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def forward(self, user_emb, entity_emb, latent_emb, latent_div_emb, edge_index, edge_type,
                interact_mat, entity_cate_set, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = _edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = _sparse_dropout(interact_mat, self.node_dropout_rate)

        """Devoted Interest Branch"""
        tmp_entity = entity_res_emb = entity_emb  # [n_entity, channel]
        tmp_user = user_res_emb = user_emb  # [n_users, channel]
        latent_res_emb = latent_emb
        for i in range(len(self.convs)):
            # (Eq. 9)
            entity_emb, user_emb, latent_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, entity_cate_set)

            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            latent_res_emb = F.normalize(latent_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)  # (Eq. 10)
            user_res_emb = torch.add(user_res_emb, user_emb)        # (Eq. 10)
            latent_res_emb = torch.add(latent_res_emb, latent_emb)  # (Eq. 3)

        """Diverse Interest Branch"""
        entity_d_emb = entity_emb = tmp_entity  # [n_entity, channel]
        user_d_emb = user_emb = tmp_user  # [n_users, channel]
        latent_d_emb = latent_div_emb
        for i in range(len(self.extends)):
            # (Eq. 9)
            entity_emb, user_emb, latent_div_emb = self.extends[i](entity_emb, user_emb, latent_div_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight_d, entity_cate_set)

            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            latent_div_emb = F.normalize(latent_div_emb)

            """result emb"""
            entity_d_emb = torch.add(entity_d_emb, entity_emb)      # (Eq. 10)
            user_d_emb = torch.add(user_d_emb, user_emb)            # (Eq. 10)
            latent_d_emb = torch.add(latent_d_emb, latent_div_emb)  # (Eq. 3)

        """message dropout"""
        if mess_dropout:
            entity_res_emb = self.dropout(entity_res_emb)
            user_res_emb = self.dropout(user_res_emb)
            latent_res_emb = self.dropout(latent_res_emb)
            entity_d_emb = self.dropout(entity_d_emb)
            user_d_emb = self.dropout(user_d_emb)
            latent_d_emb = self.dropout(latent_d_emb)
        
        # (Eq. 4)
        cor = self._cor(latent_res_emb)
        cor_d = self._cor(latent_d_emb)

        return entity_res_emb, user_res_emb, cor, entity_d_emb, user_d_emb, cor_d


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, entity_cate_set, edge_index, edge_type, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.n_cates = data_config['n_cates']

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind

        self.entity_cate_set = entity_cate_set
        self.interact_mat = adj_mat
        self.edge_index = edge_index
        self.edge_type = edge_type

        self._init_weight()
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))
        self.latent_div_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)             # devoted interest
        self.latent_div_emb = nn.Parameter(self.latent_div_emb)     # diverse interest

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_cates=self.n_cates,
                         n_factors=self.n_factors,
                         interact_mat=self.interact_mat,
                         emb_size=self.emb_size,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        div_item = batch['div_items']
        div_neg_item = batch['div_neg_items']
        user_alpha = batch['user_alpha']

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, cor, \
        entity_div_emb, user_div_emb, cor_d = self.gcn(user_emb,
                                                     entity_emb,
                                                     self.latent_emb,
                                                     self.latent_div_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     self.entity_cate_set,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)


        u_e = user_gcn_emb[user]
        u_d = user_div_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        div_e, div_nge_e = entity_div_emb[div_item], entity_div_emb[div_neg_item]

        return self.create_loss(u_e, pos_e, neg_e, cor, u_d, div_e, div_nge_e, cor_d, user_alpha)


    def generate(self, user_div_score):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb, cor,\
        entity_div_emb, user_div_emb, cor_d = self.gcn(user_emb,
                                        entity_emb,
                                        self.latent_emb,
                                        self.latent_div_emb,
                                        self.edge_index,
                                        self.edge_type,
                                        self.interact_mat,
                                        self.entity_cate_set,
                                        mess_dropout=False, node_dropout=False)

        return (entity_gcn_emb + entity_div_emb), \
               (user_div_score.unsqueeze(-1)*user_gcn_emb + (1-user_div_score).unsqueeze(-1)*user_div_emb) # (Eq. 11)


    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())     # (Eq. 12)

    # (Eq. 13)
    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                    + torch.norm(pos_items) ** 2
                    + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss + cor_loss, mf_loss, emb_loss, cor

        
    def create_loss(self, users, pos_items, neg_items, cor, users_d, div_items, div_negs, cor_d, user_alpha):

        _loss, mf_loss, emb_loss, cor = self.create_bpr_loss(users, pos_items, neg_items, cor)
        d_loss, dpp_loss, emb_d_loss, d_cor = self.create_bpr_loss(users_d, div_items, div_negs, cor_d)

        # (Eq. 14)
        loss = (user_alpha*_loss).mean() + ((1-user_alpha)*d_loss).mean() 
        
        return loss, mf_loss+dpp_loss, emb_loss+emb_d_loss, cor+d_cor
