import pandas as pd
import numpy as np
import tqdm
import argparse
from sklearn.metrics import roc_auc_score
import src.models.model as Model
import src.models.creat_data as Data

import torch
import torch.nn as nn
import torch.utils.data

def get_model(model_name, feature_nums, field_nums, latent_dims):
    if model_name == 'LR':
        return Model.LR(feature_nums)
    elif model_name == 'FM':
        return Model.FM(feature_nums, latent_dims)
    elif model_name == 'FFM':
        return Model.FFM(feature_nums, field_nums, latent_dims)

def get_dataset(datapath, dataset_name, campaign_id, valid_day, test_day):
    data_path = datapath + dataset_name + campaign_id
    data_file_name = 'train.txt'
    day_index_file_name = 'day_index.csv'

    train_fm = pd.read_csv(data_path + data_file_name, header=None).values.astype(int)

    field_nums = len(train_fm[0, 1:]) # 特征域的数量
    feature_nums = np.max(train_fm[:, 1:].flatten()) + 1 # 特征数量

    day_indexs = pd.read_csv(data_path + day_index_file_name, header=None).values
    days = day_indexs[:, 0] # 数据集中有的日期
    days_list = days.tolist()
    days_list.pop(days_list.index(valid_day))
    days_list.pop(days_list.index(test_day))

    train_data = np.array([])
    for i, day in enumerate(days_list): # 生成训练集
        current_day_index = day_indexs[days == day]
        data_index_start = current_day_index[0, 1]
        data_index_end = current_day_index[0, 2] + 1

        data_ = train_fm[data_index_start: data_index_end, :]
        if i == 0:
            train_data = data_
        else:
            train_data = np.concatenate((train_data, data_), axis=0)

    # 生成验证集
    valid_day_index = day_indexs[days == valid_day]
    valid_index_start = valid_day_index[0, 1]
    valid_index_end = valid_day_index[0, 2] + 1
    valid_data = train_fm[valid_index_start: valid_index_end, :]

    # 生成测试集
    test_day_index = day_indexs[days == test_day]
    test_index_start = test_day_index[0, 1]
    test_index_end = test_day_index[0, 2] + 1
    test_data = train_fm[test_index_start: test_index_end, :]

    return train_data, valid_data, test_data, field_nums, feature_nums
    
def train(model, optimizer, data_loader, loss, device, log_interval = 1000):
    model.train() # 转换为训练模式
    total_loss = 0
    for i, (features, labels) in enumerate(tqdm.tqdm(data_loader, smoothing = 0, mininterval = 1.0)):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
        y = model(features)
        train_loss = loss(y, labels.float())

        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()
        if (i + 1) % log_interval == 0:
            print('average loss: {}'.format(total_loss / log_interval))
            total_loss = 0

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for features, labels in tqdm.tqdm(data_loader, smoothing = 0, mininterval = 1.0):
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            y = model(features)
            targets.extend(labels.tolist()) # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def main(data_path, dataset_name, campaign_id, valid_day, test_day, latent_dims, model_name, epoch, learning_rate, weight_decay, batch_size, device):
    device = torch.device(device) # 指定运行设备

    train_data, valid_data, test_data, field_nums, feature_nums = get_dataset(data_path, dataset_name, campaign_id, valid_day, test_day)

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    valid_dataset = Data.libsvm_dataset(valid_data[:, 1:], valid_data[:, 0])
    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    model = get_model(model_name, feature_nums, field_nums, latent_dims).to(device)

    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, loss, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc', auc)
    auc = test(model, test_data_loader, device)
    print('test auc:', auc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--valid_day', default=11, help='6, 7, 8, 9, 10, 11, 12')
    parser.add_argument('--test_day', default=12, help='6, 7, 8, 9, 10, 11, 12')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3386')
    parser.add_argument('--model_name', default='FM')
    parser.add_argument('--latent_dims', default=5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', default='cpu:0')

    args = parser.parse_args()
    main(
        args.data_path,
        args.dataset_name,
        args.campaign_id,
        args.valid_day,
        args.test_day,
        args.latent_dims,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.weight_decay,
        args.batch_size,
        args.device)