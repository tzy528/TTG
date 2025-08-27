import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import LightGCN
from torch_geometric.data import Data

# 检查是否有GPU可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # 加载MovieLens 100k数据集
# data = pd.read_csv('./datasets/ML100K/train.txt', sep='\t', names=['user_id', 'item_id'])
#
# # 创建user_item_interactions矩阵
# num_users = int(max(data['user_id']))
# num_items = int(max(data['item_id']))
# user_item_interactions = np.zeros((num_users, num_items))
# for row in data.itertuples():
#     user_item_interactions[row.user_id - 1, row.item_id - 1] = 1
# user_item_interactions = torch.tensor(user_item_interactions, dtype=torch.float32).to(device)
# # 创建edge_index
# edges = []
# for row in data.itertuples():
#     edges.append([row.user_id - 1, num_users + row.item_id - 1])
# edge_index = torch.tensor(edges).t().contiguous().to(device)
#
#
# # 创建PyTorch Geometric Data对象
# graph_data = Data(edge_index=edge_index).to(device)

# 定义LightGCN模型
class LightGCNEncoder(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCNEncoder, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # print(self.num_users,self.num_items,self.embedding_dim,embedding_dim)
        self.model = LightGCN(num_users + num_items, embedding_dim, num_layers)
        # print(dir(self.model))
    def forward(self, data):
        # user_item_embeddings = self.model(data.edge_index)
        embs=self.model.get_embedding(data.edge_index)
        # print(embs.shape)
        user_embeddings = embs[:self.num_users,:]
        item_embeddings = embs[self.num_users:,:]
        return user_embeddings, item_embeddings


# 定义BPR损失函数
    def bpr_loss(self,user_embeddings, pos_item_embeddings, neg_item_embeddings):
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(neg_item_embeddings * pos_item_embeddings, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss


# 定义训练过程
# def train(model, data, num_epochs, lr):
#     model.to(device)  # 将模型移动到GPU
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         user_embeddings, item_embeddings = model(data)
#         print("Training - User Embeddings Shape:", user_embeddings.shape)  # 添加打印语句
#         print("Training - Item Embeddings Shape:", item_embeddings.shape)  # 添加打印语句
#         # 获取点击279号项目的用户嵌入
#         posuser = user_embeddings[torch.where(user_item_interactions[:, 278] > 0)[0]].to(device)
#
#         # 获取正样本（项目279的嵌入）
#         pos_item_embeddings = item_embeddings[278,:].to(device)
#
#         neguser= user_embeddings[torch.where(user_item_interactions[:, 278] == 0)[0]].to(device)
#
#         random_indices = torch.randint(0, neguser.size(0), (posuser.shape[0],))
#         neg_samples = neguser[random_indices].to(device)
#
#         # 计算BPR损失
#         loss = bpr_loss(posuser, pos_item_embeddings, neg_samples)
#         loss.backward()
#         optimizer.step()
#
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
#
#
# def decoder():
#     encoder.eval()
#     with torch.no_grad():
#         # 获取用户和项目的嵌入
#         user_embeddings, item_embeddings = encoder(graph_data)
#
#
#         # 获取点击279号项目的用户嵌入
#         targetUser = user_embeddings[torch.where(user_item_interactions[:, 278] > 0)[0]]
#
#
#         print("Embeddings for users who clicked on item 279:", targetUser.shape)
#
#
# # 初始化并训练模型
# embedding_dim = 128
# num_layers = 3
# num_epochs = 200
# lr = 0.001
#
# encoder = LightGCNEncoder(num_users, num_items, embedding_dim, num_layers)
# train(encoder, graph_data, num_epochs, lr)
# decoder()