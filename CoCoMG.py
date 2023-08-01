import os
from models.embedder import embedder
from tqdm import tqdm
from evaluate import evaluate
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.Layers import make_mlplayers
from utils.process import get_clones,get_feature_dis,get_A_r,local_preserve,CCASSL


class CoCoMG_model(nn.Module):
    def __init__(self, n_in, view_num, cfg=None, dropout=0.2, sparse=True):
        super(CoCoMG_model, self).__init__()
        self.view_num = view_num
        MLP = make_mlplayers(n_in, cfg, batch_norm=False)
        self.MLP_list = get_clones(MLP, self.view_num)
        self.dropout = dropout
        self.A = None
        self.sparse = sparse
        self.cfg = cfg

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adj_list=None):
        if self.A is None:
            self.A = adj_list

        view_num = len(adj_list)

        x_list = [F.dropout(x, self.dropout, training=self.training) for i in range(view_num)]

        z_list = [self.MLP_list[i](x_list[i]) for i in range(view_num)]

        s_list = [get_feature_dis(z_list[i]) for i in range(view_num)]

        # simple average
        z_unsqu = [z.unsqueeze(0) for z in z_list]
        z_fusion = torch.mean(torch.cat(z_unsqu), 0)

        return z_list, s_list, z_fusion

    def embed(self, x, adj_list=None):

        z_list = [self.MLP_list[i](x) for i in range(self.view_num)]

        # simple average
        z_unsqu = [z.unsqueeze(0) for z in z_list]
        z_fusion = torch.mean(torch.cat(z_unsqu), 0)

        return z_fusion.detach()


class CoCoMG(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):

        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]

        adj_label_list = [get_A_r(adj, self.args.A_r) for adj in adj_list]

        N = features.size(0)
        I_target = torch.tensor(np.eye(self.cfg[-1])).to(self.args.device)

        model = CoCoMG_model(self.args.ft_size, self.args.view_num, cfg=self.cfg, dropout=self.args.dropout).to(
            self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        model.train()
        for epoch in tqdm(range(self.args.nb_epochs)):
            model.train()
            optimiser.zero_grad()
            z_list, s_list, z_fusion = model(features, adj_list)

            loss_local = 0
            for i in range(self.args.view_num):
                loss_local += 1 * local_preserve(s_list[i], adj_label_list[i], tau=self.args.tau)

            loss_C, loss_simi = CCASSL(z_list, N, I_target, self.args.view_num)
            loss = (1 - loss_simi + loss_C * self.args.w_c) + loss_local * self.args.w_l

            loss.backward()
            optimiser.step()

        model.eval()
        hf = model.embed(features, adj_list)
        acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, nmi, nmi_std, acc_cluster, acc_cluster_std, st = evaluate(
            hf, self.idx_train, self.idx_val, self.idx_test, self.labels,
            epoch=self.args.test_epo, lr=self.args.test_lr)
        return acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, nmi, nmi_std, acc_cluster, acc_cluster_std, st


