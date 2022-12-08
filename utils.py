import numpy as np
import scipy.sparse as sp
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos

def knn1(feature, k):
    adj = np.zeros((len(feature), len(feature)), dtype=np.int64)
    dist = cos(feature.detach().numpy())
   # dist =dist - np.array(torch.eye(len(feature)))
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    row = np.arange(len(feature)).repeat(k + 1)
    #print(len(col))
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)),
                        shape=(len(feature), len(feature)),
                        dtype=np.float32)
    return adj

def knn(feature, t):
    dist = torch.tensor(cos(feature.detach().numpy()))
    edge = torch.nonzero(dist > t).squeeze().tolist()
    edge = list(map(list, zip(*edge)))
    row = np.array(edge[0])
    col = np.array(edge[1])
    print(len(col))
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)),
                        shape=(len(feature), len(feature)),
                        dtype=np.float32)
    return adj

def load_data(dataset, K):
    """Load citation network dataset (cora only for now)"""
    print("loading data.....")
    aff = torch.tensor(np.load(dataset+"aff.npy"))
    pic_feature =np.load(dataset + "pic_feature.npy")
    text_feature = np.load(dataset + "text_feature.npy")

    scores = torch.tensor(cos(torch.tensor(text_feature).detach().numpy()))
    scores = scores - torch.diag_embed(torch.diag(scores))
    ind = torch.nonzero(scores > 0.9).squeeze().tolist()
    print("before text:",len(ind))

    scores = torch.tensor(cos(torch.tensor(pic_feature).detach().numpy()))
    scores = scores - torch.diag_embed(torch.diag(scores))
    ind = torch.nonzero(scores > 0.9).squeeze().tolist()
    print("before pic:", len(ind))

    print("generating graph...")
    adj0 = knn(torch.tensor(pic_feature),0.95)
    adj1 = knn(torch.tensor(text_feature), 0.95)

    adj0 = normalize(adj0)  #AXW   D-1*A or D(-1/2)*A*D(-1/2)
    adj1 = normalize(adj1)

    adj0 = sparse_mx_to_torch_sparse_tensor(adj0)
    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)

    pic_feature = sp.csr_matrix(pic_feature, dtype=np.float32)
    #print(pic_feature)
    pic_feature = normalize(pic_feature)
    pic_feature = torch.FloatTensor(np.array(pic_feature.todense()))
    #print(pic_feature)

    text_feature = sp.csr_matrix(text_feature, dtype=np.float32)
    text_feature = normalize(text_feature)
    text_feature = torch.FloatTensor(np.array(text_feature.todense()))

    trainset=aff[:int(len(aff)*0.7)]
    testset = aff[int(len(aff) * 0.7):]
    print("load finishing!...")

    return adj0, adj1,  pic_feature,  text_feature, trainset, testset

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)