import torch
import numpy as np
import torch.nn as nn
import copy
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import cosine_similarity

BASEPATH = 'utils/data/' # replace your path of datasets

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj
    return adj_label


def local_preserve(x_dis, adj_label, tau=1.0):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


def CCASSL(z_list, N, I_target, num_view):
    if num_view == 2:
        embeding_a = z_list[0]
        embeding_b = z_list[1]
        embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
        embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)
        c1 = torch.mm(embeding_a.T, embeding_a) / N
        c2 = torch.mm(embeding_b.T, embeding_b) / N
        loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
        loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
        loss_C = loss_c1 + loss_c2
        loss_simi = cosine_similarity(embeding_a, embeding_b, dim=-1).mean()
    else:
        embeding_a = z_list[0]
        embeding_b = z_list[1]
        embeding_c = z_list[2]
        embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
        embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)
        embeding_c = (embeding_c - embeding_c.mean(0)) / embeding_c.std(0)
        c1 = torch.mm(embeding_a.T, embeding_a) / N
        c2 = torch.mm(embeding_b.T, embeding_b) / N
        c3 = torch.mm(embeding_c.T, embeding_c) / N
        loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
        loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
        loss_c3 = (I_target - c3).pow(2).mean() + torch.diag(c3).mean()
        loss_C = loss_c1 + loss_c2 + loss_c3
        loss_simi = cosine_similarity(embeding_a, embeding_b, dim=-1).mean() + cosine_similarity(embeding_a, embeding_c,
                                                                                                 dim=-1).mean() + cosine_similarity(
            embeding_b, embeding_c, dim=-1).mean()
    return loss_C, loss_simi


def get_feature_dis(x, eps=1e-8):
    """
    x :           batch_size x nhid
    eps: small value to avoid division by 0 default 1e-8
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.clamp(x_sum, min=eps)  # clamp to avoid division by zero
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


def load_acm_mat(sc=3):
    data = sio.loadmat(BASEPATH + 'acm.mat')
    label = data['label']

    adj_edge1 = data["PLP"]
    adj_edge2 = data["PAP"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0]) * sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_dblp4057(sc=3):
    data = sio.loadmat(BASEPATH + 'dblp4057.mat')
    label = data['label']

    adj_edge1 = data["net_APCPA"]
    adj_edge2 = data["net_APA"]
    adj_edge3 = data["net_APTPA"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = adj_edge1 + np.eye(adj_edge1.shape[0]) * sc
    adj2 = adj_edge2 + np.eye(adj_edge2.shape[0]) * sc
    adj3 = adj_edge3 + np.eye(adj_edge3.shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * sc

    truefeatures = data['features'].astype(float)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_list = [adj1, adj2, adj3]
    adj_fusion = sp.csr_matrix(adj_fusion)
    truefeatures = sp.lil_matrix(truefeatures)

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_imdb4780(sc=3):
    data = sio.loadmat(BASEPATH + 'imdb4780.mat')
    label = data['label']
    ###########################################################
    adj_edge1 = data["MDM"]
    adj_edge2 = data["MAM"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1
    ############################################################
    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0]) * sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * sc

    truefeatures = data['feature'].astype(float)

    truefeatures = sp.lil_matrix(truefeatures)

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)
    adj_list = [adj1, adj2]

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_freebase(sc=3):
    type_num = 3492
    ratio = [20, 40, 60]
    # The order of node types: 0 m 1 d 2 a 3 w
    path = BASEPATH + "freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.eye(type_num)

    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")

    adj_fusion1 = np.array(mam.todense(), dtype=int) + np.array(mdm.todense(), dtype=int) + np.array(mwm.todense(),
                                                                                                     dtype=int)
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * sc

    adj1 = sp.csr_matrix(mam)
    adj2 = sp.csr_matrix(mdm)
    adj3 = sp.csr_matrix(mwm)
    adj_fusion = sp.csr_matrix(adj_fusion)

    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.FloatTensor(label)
    adj_list = [adj1, adj2, adj3]

    # pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    return adj_list, feat_m, label, train[0], val[0], test[0], adj_fusion


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features


def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
