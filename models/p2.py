import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import LightGCN
from torch_geometric.data import Data
import torch.nn.functional as F




# ����LightGCNģ��
class LightGCNPredictor(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCNPredictor, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # print(self.num_users,self.num_items,self.embedding_dim,embedding_dim)
        self.model = LightGCN(num_users + num_items, embedding_dim, num_layers)
        # print(dir(self.model))

    def forward(self, data, src):
        # ��ȡ�û�����Ŀ��Ƕ��
        embs = self.model.get_embedding(data.edge_index)
        # ��ȡĿ���û���Ƕ��
        user_embeddings = embs[:self.num_users,:]
        item_embeddings = embs[self.num_users:,:]
        user_emb_src = user_embeddings[src]
        # �����û�-��Ŀ��������
        scores = torch.matmul(user_emb_src, item_embeddings.T)
        return scores  # ����Ԥ����� [batch_size, num_items]

    def RecLoss(self, scores, target_items):
        '''
        scores: ģ�͵�Ԥ����� [batch_size, num_items]
        tarlist: Ŀ����Ŀ�б����� [item1_idx, item2_idx, ...]
        '''
        # # ����Ŀ���ǩ��tarlist�е���Ŀ��ǩΪ1������Ϊ0
        target = torch.zeros_like(scores)
        target[:, target_items] = 1
        # # ʹ�ö�Ԫ��������ʧ�����ǩ���ࣩ
        loss = F.binary_cross_entropy_with_logits(scores, target)

        # target_items = torch.tensor(target_items, dtype=torch.long).to(scores.device)
        # Compute negative log likelihood loss
        # loss = F.cross_entropy(scores, target)
        # target_items = target_items.unique().long().to(scores.device)

        return loss

    # def RecLoss(self,toplist,tarlist,taruser):
    #
    #     tarlist_one_hot = torch.zeros_like(toplist, dtype=torch.float32)
    #     for tar in tarlist:
    #         tarlist_one_hot += (toplist == tar).float()
    #
    #     # ����Ŀ������ toplist �е��ܳ��ִ���
    #     total_count = tarlist_one_hot.sum()
    #
    #     # ������ʧ
    #     denominator = len(tarlist) * taruser.shape[0]
    #     recloss = 1 - total_count / denominator
    #
    #     return recloss
    def recommend(self,data,src_index,k):
        # �����û�-��Ŀ��������
        toplist=self.model.recommend(data.edge_index,src_index=src_index,k=k)
        # �����Ƽ���Ŀ������
        return toplist

    def getemb(self, data):
        # user_item_embeddings = self.model(data.edge_index)
        embs = self.model.get_embedding(data.edge_index)
        # print(embs.shape)
        user_embeddings = embs[:self.num_users, :]
        item_embeddings = embs[self.num_users:, :]
        return user_embeddings, item_embeddings

        # ����BPR��ʧ����

    def bpr_loss(self, user_embeddings, pos_item_embeddings, neg_item_embeddings):
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(neg_item_embeddings * pos_item_embeddings, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss