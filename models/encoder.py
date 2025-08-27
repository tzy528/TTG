import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_indices, item_indices, adj_mat):
        # all_embeddings = []
        # user_emb = self.user_embedding(user_indices)
        # item_emb = self.item_embedding(item_indices)
        # all_embeddings.append(torch.cat([user_emb, item_emb], dim=0))
        #
        # for _ in range(self.num_layers):
        #     all_emb = torch.cat([user_emb, item_emb], dim=0)
        #     print(adj_mat.shape,all_emb.shape)
        #     all_emb = torch.sparse.mm(adj_mat, all_emb)
        #
        #     user_emb, item_emb = torch.split(all_emb, [self.num_users, self.num_items], dim=0)
        #     all_embeddings.append(all_emb)
        #
        # final_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)
        # user_emb, item_emb = torch.split(final_embeddings, [self.num_users, self.num_items], dim=0)
        users_embeds = self.user_embedding.weight
        items_embeds = self.item_embedding.weight
        all_users, all_items = users_embeds, items_embeds
        adj_mat = torch.matmul(users_embeds, items_embeds.T) / (self.user_embedding.num_embeddings)

        for _ in range(self.num_layers):
            all_users = torch.matmul(adj_mat, all_items)
            all_items = torch.matmul(adj_mat.T, all_users)

        return all_users, all_items


class BPRLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super(BPRLoss, self).__init__()
        self.sigma = sigma

    def forward(self, pos_score, neg_score):
        diff = pos_score - neg_score
        loss = -torch.log(torch.sigmoid(diff / self.sigma))
        return torch.mean(loss)



