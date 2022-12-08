import torch.nn as nn
import torch.nn.functional as F
from code.layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
from code.parser import parse_args
import random
from code.contrastive import Contrast_L, Contrast_PR
from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
import scipy.sparse as sp

class GNN(nn.Module):
    def __init__(self, input, hidden, output, dropout, baiss, args):
        super(GNN, self).__init__()
        self.input = input
        self.hidden = hidden
        self.output = output
        self.baiss = baiss
        self.gc01 = GraphConvolution(input, hidden, baiss)
        self.gc02 = GraphConvolution(hidden, output, baiss)

        self.gc11 = GraphConvolution(input, hidden, baiss)
        self.gc12 = GraphConvolution(hidden, output, baiss)

        self.linear = nn.Linear(hidden * 2, hidden)
        self.lear_p = nn.Linear(output * 4, output * 2)
        self.lear_r = nn.Linear(output * 4, output * 2)
        self.lear_neg = nn.Linear(output * 4, output * 2)

        self.t = args.t
        self.L = args.L
        self.gamma = args.gamma
        self.walk_n = args.walk_n
        self.lc_sample = args.lc_sample
        self.tau0 = args.tau_0
        self.tau1 = args.tau_1

        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2

        self.lambda_super = args.lambda_super

        self.K = args.K

        self.alfa = args.alfa
        self.pseudo_n = args.pseudo_n

        self.dropout = dropout
        self.l2 = args.l2

    def forward(self, x0, adj0, x1, adj1, trainset):
        x0 = self.gc01(x0, adj0)
        #x0 = F.dropout(x0, self.dropout, training=self.training)
        #x0 = self.gc02(x0, adj0)
        #print("1")
        #x0 = F.log_softmax(x0, dim=1)
        x0 = x0 / torch.norm(x0,p=2,dim=1).unsqueeze(1)

        x1 = self.gc11(x1, adj1)
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        #x1 = self.gc12(x1, adj1)
        #F.log_softmax(x, dim=1)
        #x1 = F.log_softmax(x1, dim=1)
        x1 = x1 / torch.norm(x1, p=2, dim=1).unsqueeze(1)

        x_all = torch.cat([x0, x1], dim=1)#output*2
        #print(x_all.shape)
        #x_all = F.relu(self.linear(x_all))
        #x_all = x0

        lc_loss = self.lc_loss(adj0, adj1, x_all)
        print(lc_loss)

        # real
        real = torch.tensor(list(map(list, zip(*trainset.tolist()))))

        r_s_emb = x_all[real[0]]
        r_e_emb = x_all[real[1]]
        neg_index0 = torch.tensor(random.sample(range(0, len(x_all)-1), len(r_s_emb)))
        neg_r = x_all[neg_index0]
        #pseudo
        scores = torch.tensor(cos(torch.tensor(x_all).detach().numpy()))
        scores = scores - torch.diag_embed(torch.diag(scores))
        scores = torch.triu(scores,diagonal=0)
        ind = torch.nonzero(scores > 0.95).squeeze().tolist()
        print(len(ind))
        ind = torch.tensor(list(map(list, zip(*ind))))
        start = ind[0]
        end = ind[1]
        p_s_emb = x_all[start]
        p_e_emb = x_all[end]
        neg_index1 = torch.tensor(np.random.choice(range(0, len(x_all) - 1), len(p_s_emb)))
        neg_p = x_all[neg_index1]

        s_emb = torch.cat([r_s_emb, p_s_emb], dim=0)
        e_emb = torch.cat([r_e_emb, p_e_emb], dim=0)
        neg = torch.cat([neg_r, neg_p], dim=0)

        #loss_label = self.bpr_loss1(s_emb, e_emb, neg)
        loss_label = self.bpr_loss2(s_emb, e_emb, neg)
        #print(loss_label)
        """
        adj0_new = self.knn(x0, self.K)
        adj1_new = self.knn(x1, self.K)

        adj0_new = self.normalize(adj0_new)
        adj1_new = self.normalize(adj1_new)

        adj0_new = self.sparse_mx_to_torch_sparse_tensor(adj0_new)
        adj1_new = self.sparse_mx_to_torch_sparse_tensor(adj1_new)

        loss_G = torch.norm(adj0_new - adj0 + adj1_new - adj1)
        """
        #return x_all, adj0_new, adj1_new, lc_loss + self.lambda_1 * loss_label + self.lambda_2 * loss_G
        return x_all, loss_label + self.lambda_1 * lc_loss
        #return x_all, loss_label

    def lc_loss(self, adj0, adj1, x_all):
        walk_n = self.walk_n
        L = self.L
        gamma = self.gamma
        n_list = torch.tensor(sorted(random.sample(range(0,len(x_all)),self.lc_sample)))

        node_a, node_b, nebor_L = self.walk(adj0, adj1, n_list, L, walk_n)
        #print(len(node_a), len(node_b), len(nebor_L))
        x_0 = x_all[node_a]
        x_1 = x_all[node_b]

        contrastive_L = Contrast_L(self.tau0)
        lc = contrastive_L.forward(x_0, x_1, nebor_L, gamma)
        return lc

    def walk(self, adj1, adj2, n_list, L, walk_n):
        node_a = torch.LongTensor([])
        node_b = torch.LongTensor([])
        nebor_L = torch.LongTensor([])
        c = 0
        for node in n_list:
            for j in range(walk_n):
                nebor, layer = self.nebor_sam(adj1 + adj2, L, node)
                tmp_node = node.repeat(len(nebor))
                node_a = torch.cat([node_a, tmp_node], dim=0)
                node_b = torch.cat([node_b, nebor], dim=0)
                nebor_L = torch.cat([nebor_L, layer], dim=0)
        return node_a, node_b, nebor_L

    def nebor_sam(self, adj, L, node):
        node_list=[]
        layer=[]
        rm = []
        adj = adj.to_dense() - torch.diag_embed(torch.diag(adj.to_dense()))
        for i in range(1, L+1):
            a = torch.nonzero(adj[node]).squeeze(1).tolist()  #所有邻居
            #b = rm
            nebor = torch.tensor(list(set(a).difference(set(rm))))  #和上一次的邻居去重
            rm = rm + a
            rm = list(set(rm))
            if len(nebor) == 0:
                break
            else:
                nn = nebor[random.randint(0,len(nebor)-1)]
                node_list.append(int(nn))
                layer.append(i)
                node = nn
        return torch.LongTensor(node_list), torch.LongTensor(layer)

    def bpr_loss1(self, x_0, x_1, x_neg):
        pos_scores = F.cosine_similarity(x_0, x_1, dim=-1)
        neg_scores = F.cosine_similarity(x_0, x_neg, dim=-1)
        mf_loss = torch.log(1+torch.exp(neg_scores - pos_scores).sum(dim=0))
        #print("loss:",mf_loss)
        return mf_loss

    def bpr_loss2(self, x_0, x_1, x_neg):
        pos_scores = F.cosine_similarity(x_0, x_1, dim=-1)
        neg_scores = F.cosine_similarity(x_0, x_neg, dim=-1)
        weight = ((pos_scores-self.t)/(1-self.t))**self.alfa
        second = torch.log(1+torch.exp(neg_scores - pos_scores))
        mf_loss = torch.sum(torch.mul(weight, second), axis=0)
        return mf_loss

    def knn1(self, feature, k):
        adj = np.zeros((len(feature), len(feature)), dtype=np.int64)
        dist = cos(feature.detach().numpy())
        # dist =dist - np.array(torch.eye(len(feature)))
        col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
        row = np.arange(len(feature)).repeat(k + 1)
        adj = sp.coo_matrix((np.ones(len(row)), (row, col)),
                            shape=(len(feature), len(feature)),
                            dtype=np.float32)
        return adj

    def knn(self, feature, K):
        dist = torch.tensor(cos(feature.detach().numpy()))
        # dist = dist - np.array(torch.eye(len(feature)))
        # print(dist)
        edge = torch.nonzero(dist > 0.85).squeeze().tolist()
        edge = list(map(list, zip(*edge)))
        row = np.array(edge[0])
        col = np.array(edge[1])
        adj = sp.coo_matrix((np.ones(len(row)), (row, col)),
                            shape=(len(feature), len(feature)),
                            dtype=np.float32)
        return adj

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def generate(self, x0, adj0, x1, adj1):
        x0 = self.gc01(x0, adj0)
        # x0 = F.dropout(x0, self.dropout, training=self.training)
        # x0 = self.gc02(x0, adj0)
        # print("1")
        # x0 = F.log_softmax(x0, dim=1)
        x0 = x0 / torch.norm(x0, p=2, dim=1).unsqueeze(1)

        x1 = self.gc11(x1, adj1)
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        # x1 = self.gc12(x1, adj1)
        # F.log_softmax(x, dim=1)
        # x1 = F.log_softmax(x1, dim=1)
        x1 = x1 / torch.norm(x1, p=2, dim=1).unsqueeze(1)

        x_all = torch.cat([x0, x1], dim=1)  # output*2
        return x_all