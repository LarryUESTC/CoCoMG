import argparse


class Unsup_CoCoMG():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
        self.parser.add_argument('--sparse_adj', type=bool, default=False, help='sparsity of adjacency matrix')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        self.parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the models')
        self.parser.add_argument('--cfg', type=int, default=[512, 128], help='hidden dimension')
        self.parser.add_argument('--nb_epochs', type=int, default=400, help='the number of epochs')
        self.parser.add_argument('--test_epo', type=int, default=100, help='test_epo')
        self.parser.add_argument('--test_lr', type=int, default=0.01, help='test_lr')
        self.parser.add_argument('--sc', type=int, default=0, help='')
        self.parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
        self.parser.add_argument('--w_c', type=float, default=0.01, help='weight of loss loss_C')
        self.parser.add_argument('--w_l', type=float, default=1, help='weight of loss loss_local')
        self.parser.add_argument('--tau', type=float, default=1.0, help='weight of tau')
        self.parser.add_argument('--A_r', type=int, default=2, help='weight of A_r')

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        pass

    def get_parse(self):
        return self.args


class Unsup_CoCoMG_Acm(Unsup_CoCoMG):
    def __init__(self):
        super(Unsup_CoCoMG_Acm, self).__init__()
        self.parser.add_argument('--view_num', type=int, default=2, help='view number')
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_CoCoMG_Acm, self).replace()
        self.args.__setattr__('dataset', 'acm')
        self.args.__setattr__('cfg', [256, 256, 128, 128])
        self.args.__setattr__('lr', 0.005)
        self.args.__setattr__('nb_epochs', 200)
        self.args.__setattr__('test_epo', 100)
        self.args.__setattr__('test_lr', 0.01)
        self.args.__setattr__('w_c', 0.1)
        self.args.__setattr__('w_l', 5)
        self.args.__setattr__('tau', 1.0)
        self.args.__setattr__('A_r', 3)
        self.args.__setattr__('dropout', 0.5)
        self.args.__setattr__('seed', 2022)


class Unsup_CoCoMG_Imdb4780(Unsup_CoCoMG):
    def __init__(self):
        super(Unsup_CoCoMG_Imdb4780, self).__init__()
        self.parser.add_argument('--view_num', type=int, default=2, help='view number')
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_CoCoMG_Imdb4780, self).replace()
        self.args.__setattr__('dataset', 'imdb4780')
        self.args.__setattr__('cfg', [256, 128, 128])
        self.args.__setattr__('lr', 0.01)
        self.args.__setattr__('nb_epochs', 500)
        self.args.__setattr__('test_epo', 100)
        self.args.__setattr__('test_lr', 0.01)
        self.args.__setattr__('w_c', 1)
        self.args.__setattr__('w_l', 1)
        self.args.__setattr__('tau', 1)
        self.args.__setattr__('A_r', 3)
        self.args.__setattr__('dropout', 0.3)
        self.args.__setattr__('seed', 2022)


class Unsup_CoCoMG_Dblp4057(Unsup_CoCoMG):
    def __init__(self):
        super(Unsup_CoCoMG_Dblp4057, self).__init__()
        self.parser.add_argument('--view_num', type=int, default=3, help='view number')
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_CoCoMG_Dblp4057, self).replace()
        self.args.__setattr__('dataset', 'dblp4057')
        self.args.__setattr__('cfg', [512, 256, 256, 128, 128])
        self.args.__setattr__('lr', 0.001)
        self.args.__setattr__('nb_epochs', 2000)
        self.args.__setattr__('test_epo', 100)
        self.args.__setattr__('test_lr', 0.01)
        self.args.__setattr__('w_c', 1)
        self.args.__setattr__('w_l', 10)
        self.args.__setattr__('tau', 2.0)
        self.args.__setattr__('A_r', 3)
        self.args.__setattr__('dropout', 0.1)
        self.args.__setattr__('seed', 2023)


class Unsup_CoCoMG_Freebase(Unsup_CoCoMG):
    def __init__(self):
        super(Unsup_CoCoMG_Freebase, self).__init__()
        self.parser.add_argument('--view_num', type=int, default=3, help='view number')
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_CoCoMG_Freebase, self).replace()
        self.args.__setattr__('dataset', 'freebase')
        self.args.__setattr__('cfg', [256, 256, 128])
        self.args.__setattr__('lr', 0.001)
        self.args.__setattr__('nb_epochs', 600)
        self.args.__setattr__('test_epo', 100)
        self.args.__setattr__('test_lr', 0.01)
        self.args.__setattr__('w_c', 0.00001)
        self.args.__setattr__('w_l', 10)
        self.args.__setattr__('tau', 5)
        self.args.__setattr__('dropout', 0.0)
        self.args.__setattr__('A_r', 4)
        self.args.__setattr__('seed', 2022)


def getParams(dataset):
    if dataset == "acm":
        return Unsup_CoCoMG_Acm().get_parse()
    elif dataset == "dblp4057":
        return Unsup_CoCoMG_Dblp4057().get_parse()
    elif dataset == "imdb4780":
        return Unsup_CoCoMG_Imdb4780().get_parse()
    elif dataset == "freebase":
        return Unsup_CoCoMG_Freebase().get_parse()
    else:
        raise NotImplementedError(f"no suitable param for {dataset} dataset")
