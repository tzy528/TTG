import numpy as np
from fileinput import filename
import random

import pandas
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)


"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
# from os.path import join
# import sys
# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from scipy.sparse import csr_matrix
# import scipy.sparse as sp
# import world
# from world import cprint
from time import time


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    def __init__(self,path):
        super().__init__()
        print(path)
        self.path = path
        data = pandas.read_csv(self.path, delimiter='\t',names=['user_id','item_id'])
        self.num_users = int(max(data['user_id']))
        self.num_items = int(max(data['item_id']))
        self.user_item_interactions = np.zeros((self.num_users, self.num_items))
        for row in data.itertuples():
            self.user_item_interactions[row.user_id - 1, row.item_id - 1] = 1
        self.user_item_interactions = torch.tensor(self.user_item_interactions, dtype=torch.float32)
        # ´´½¨edge_index
        self.edges = []
        for row in data.itertuples():
            self.edges.append([row.user_id - 1, self.num_users + row.item_id - 1])
        self.edge_index = torch.tensor(self.edges).t().contiguous()



    def getData(self):

        return self.num_users,self.num_items,self.user_item_interactions,self.edge_index



    def getTraindata(self,taritemID):
        poslist=[]
        neglist=[]
        list1=[]
        for uid,iidlist in self.train_dict.items():
            if taritemID in iidlist:
                poslist.append(uid)
            else:
                neglist.append(uid)
            list1.append(uid)
        print(len(list1))
        self.poslist=poslist
        train_data = []

        for u in list1:
            # ulist.append(uid)
            neg_u = np.random.choice(neglist)
            train_data.append((u-1, taritemID-1, neg_u-1))  # (user, positive_item, negative_user)
            # print(train_data)

        train_data = torch.tensor(train_data, dtype=torch.long)
        return train_data,poslist
    def getPosuser(self):
        poslist=torch.tensor(self.poslist, dtype=torch.long)
        return poslist


    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

