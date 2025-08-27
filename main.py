"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
import models.encoder as Encoder
from copy import deepcopy
from models.gcn import LightGCNEncoder as LGCencoder
from models.decoder import Decoder
from models.rec import LightGCNPredictor as Predictor
from torch_geometric.data import Data
import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp_clean/', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/', help='load data path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='1', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[256]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=5, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
# params for lightGCN
parser.add_argument('--encoder_embedding_dim', type=int, default=256,
                    help="the batch size for bpr loss training procedure")
parser.add_argument('--encoder_num_layers', type=int, default=2,
                    help="the hidden layers number of encoder")
parser.add_argument('--encoder_learning_rate', type=int, default=0.001,
                    help="the learning rate of encoder")
parser.add_argument('--encoder_num_epochs', type=int, default=150,
                    help="the epochs number of encoder")
parser.add_argument('--alpha', type=float, default=1,
                    help="loss weight of encoder")
parser.add_argument('--beta', type=float, default=1,
                    help="loss weight of decoder")
parser.add_argument('--gema', type=float, default=1,
                    help="loss weight of decoder")



args = parser.parse_args()
print("args:", args)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# device = torch.device("cuda:1" if args.cuda else "cpu")
device = torch.device('cuda:1')

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
# train_path = args.data_path +args.dataset + 'train_list.npy'
# valid_path = args.data_path +args.dataset + 'valid_list.npy'
# test_path = args.data_path +args.dataset + 'test_list.npy'

# train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
# train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))           #A:[user,[ item]]
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
# test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

train_data= "./datasets/ML100K/train.txt"
taritemID=279 #279 # 1980 #1854
traindata=data_utils.Loader(train_data)
num_users,num_items,user_item_interactions,edge_index = traindata.getData()
# user_item_interactions.to(device)
# edge_index.to(device)
taruser = torch.where(user_item_interactions[:, taritemID - 1] > 0)[0].to(device)

# 创建PyTorch Geometric Data对象
graph_data = Data(edge_index=edge_index).to(device)

print('data ready.')

### Build lightGCL encoder ###
tarlist=[5,325,648,761,259,1305,981,364,1450,1601]
# tarlist=[5,324,787,654,250,9870,3450,2160,3535,1540]
encoder = LGCencoder(num_users, num_items, args.encoder_embedding_dim, args.encoder_num_layers).to(device)
# criterion = LGCencoder.bpr_loss(posuser, pos_item_embeddings, neg_samples).to(device)
# optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_learning_rate)
decoder = Decoder(args.encoder_embedding_dim,num_items).to(device)

# optimizerD = optim.Adam(decoder.parameters(), lr=args.encoder_learning_rate)
origInter = user_item_interactions[torch.where(user_item_interactions[:, taritemID-1] > 0)[0]].to(device)
predictor=Predictor(num_users+int(origInter.shape[0]), num_items, args.encoder_embedding_dim, args.encoder_num_layers).to(device)

joinOpti = optim.Adam(list(encoder.parameters())+list(decoder.parameters())+list(predictor.parameters()), lr=args.encoder_learning_rate)
# 训练encoder
best_recall = 0
batch_count = 0
print("encoder")
for epoch in range(args.encoder_num_epochs):
    encoder.train()
    joinOpti.zero_grad()
    user_embeddings, item_embeddings = encoder(graph_data)

    # 获取点击279号项目的用户嵌入
    posuser = user_embeddings[torch.where(user_item_interactions[:, taritemID-1] > 0)[0]].to(device)

    # 获取正样本（项目279的嵌入）
    pos_item_embeddings = item_embeddings[taritemID-1,:].to(device)

    neguser= user_embeddings[torch.where(user_item_interactions[:, taritemID-1] == 0)[0]].to(device)

    random_indices = torch.randint(0, neguser.size(0), (posuser.shape[0],))
    neg_samples = neguser[random_indices].to(device)

    # 计算BPR损失
    lossE = encoder.bpr_loss(posuser, pos_item_embeddings, neg_samples)
    # print('LOss',lossE)

    decoder.train()
    out = decoder(posuser)
    x_round = torch.round(abs(out), decimals=0)
    lossD = decoder.MAEloss(origInter,x_round)
    # lossD = decoder.MAEloss(origInter, out)
    # print('LOss', lossD)

    list1=[]
    x_round = x_round.cpu()
    for i in range(x_round.shape[0]):
        for j in range(x_round.shape[1]):
            if x_round[i][j].item()==1 or j in tarlist:
                list1.append([num_users+i,num_users+j])
    edge_index_tar=torch.tensor(list1).t().contiguous().to(device)
    edge_index=edge_index.to(device)

    edge_index_tar=torch.concat([edge_index,edge_index_tar],dim=1)
    graph_data_tar = Data(edge_index=edge_index_tar).to(device)


    predictor.train()
    tarlist_1=[x + num_users for x in tarlist]
    tarlist_1=torch.tensor(tarlist_1).to(device)
    # print('tarlist',tarlist_1)

    top_scores = predictor(graph_data_tar, taruser)  # 获取分数 [batch_size, num_items]
    # print(top_scores.shape)
    # print(top_scores)
    recloss = predictor.RecLoss(top_scores, tarlist)
    # print(recloss)

    joinLoss= args.alpha * lossE + args.beta * lossD + args.gema * recloss

    toplist = predictor.recommend(graph_data_tar, taruser,1000)
    # print('toplist',toplist)
    flattened_toplist = toplist.view(-1)
    counts = [torch.sum(flattened_toplist == value).item() for value in tarlist_1]
    sumcou = sum(counts)
    recloss = sumcou / len(tarlist * taruser.shape[0])


    if recloss>=best_recall:
        best_recall=recloss
        batch_count = 0
        print(best_recall)
    else:
        batch_count+=1
        if batch_count>2:
            break


    joinLoss.backward()
    joinOpti.step()

    if (epoch + 1) % 10 == 0:
        print(recloss)
        print(f'Epoch [{epoch + 1}/{args.encoder_num_epochs}], Loss: {joinLoss.item():.8f}')

encoder.eval()
decoder.eval()
predictor.eval()

savelist1=[]
with torch.no_grad():
    # 获取用户和项目的嵌入
    user_embeddings, item_embeddings = encoder(graph_data)

    # 获取点击279号项目的用户嵌入
    targetUser = user_embeddings[torch.where(user_item_interactions[:, taritemID-1] > 0)[0]].to(device)


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build Diffusion MLP ###
out_dims = eval(args.dims) + [args.encoder_embedding_dim]
# print(out_dims)
in_dims = out_dims[::-1]
diffmodel = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

optimizer = optim.AdamW(diffmodel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")


best_recall, best_epoch = -100, 0
batch_count = 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):

    diffmodel.train()
    start_time = time.time()


    total_loss = 0.0

    optimizer.zero_grad()

    fakeUser = diffusion.getAttack(diffmodel, targetUser, args.sampling_steps, args.sampling_noise).to(device)
    # fakeUser = torch.tensor(fakeUser, dtype=torch.float32).to(device)
    out = decoder(fakeUser)
    list1 = []
    x_round = torch.round(abs(out), decimals=0).cpu()
    for i in range(x_round.shape[0]):
        for j in range(x_round.shape[1]):
            if x_round[i][j].item() == 1 or j in tarlist:
                list1.append([num_users + i, num_users + j])
    edge_index_tar = torch.tensor(list1).t().contiguous().to(device)
    edge_index = edge_index.to(device)

    edge_index_tar = torch.concat([edge_index, edge_index_tar], dim=1)
    graph_data_tar = Data(edge_index=edge_index_tar).to(device)


    toplist = predictor.recommend(graph_data_tar, taruser, 1000)
    flattened_toplist = toplist.view(-1)
    counts = [torch.sum(flattened_toplist == value).item() for value in tarlist_1]
    sumcou = sum(counts)
    recloss = sumcou / len(tarlist * taruser.shape[0])
    print(recloss)

    if epoch>5:
        if recloss>=best_recall:
            best_recall=recloss
            batch_count = 0
        else:
            batch_count+=1

            if batch_count>=2:
                savelist = []
                for i in range(out.shape[0]):
                    for j in range(out.shape[1]):
                        savelist.append([i, j, round(out[i][j].item(), 3)])
                savelist = pandas.DataFrame(savelist)
                savelist.to_csv('./output/100k/withrec' + str(args.encoder_num_epochs) + '.txt', index=None, sep='\t',
                            header=False)
                break




    losses = diffusion.training_losses(diffmodel, targetUser, args.reweight)
    loss = losses["loss"].mean()

    total_loss += loss
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{args.epochs + 1}], Loss: {loss.item():.8f}')
    '''
            torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))
    
    '''



diffmodel.eval()

with torch.no_grad():
    fakeUser = diffusion.getAttack(diffmodel, targetUser, args.sampling_steps, args.sampling_noise)
    fakeUser = torch.tensor(fakeUser, dtype=torch.float32).to(device)

savelist=[]
# decoder.eval()
with torch.no_grad():
    out = decoder(fakeUser)

    print(out)
    print(out.shape)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
          savelist.append([i,j,round(out[i][j].item(),3)])
savelist=pandas.DataFrame(savelist)
savelist.to_csv('./output/yelp/withrec'+ str(args.encoder_num_epochs) +'.txt',index=None,sep='\t',header=False)


print('done')