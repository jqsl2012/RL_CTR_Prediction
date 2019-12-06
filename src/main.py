import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
import src.models.model as Model
import src.models.creat_data as Data

import torch
import torch.nn as nn
import torch.utils.data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

    return train_fm, day_indexs, train_data, valid_data, test_data, field_nums, feature_nums
    
def train(model, optimizer, data_loader, loss, device):
    model.train() # 转换为训练模式
    total_loss = 0
    log_intervals = 0
    for i, (features, labels) in enumerate(data_loader):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
        y = model(features)
        train_loss = loss(y, labels.float())

        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item() # 取张量tensor里的标量值，如果直接返回train_loss很可能会造成GPU out of memory

        log_intervals += 1

    return total_loss / log_intervals

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            y = model(features)
            targets.extend(labels.tolist()) # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def main(data_path, dataset_name, campaign_id, valid_day, test_day, latent_dims, model_name, epoch, learning_rate, weight_decay, batch_size, device, save_param_dir):
    if not os.path.exists(save_param_dir):
        os.mkdir(save_param_dir)

    device = torch.device(device) # 指定运行设备
    train_fm, day_indexs, train_data, valid_data, test_data, field_nums, feature_nums = get_dataset(data_path, dataset_name, campaign_id, valid_day, test_day)

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    valid_dataset = Data.libsvm_dataset(valid_data[:, 1:], valid_data[:, 0])
    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    model = get_model(model_name, feature_nums, field_nums, latent_dims).to(device)

    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    valid_aucs = []
    early_stop_index = 0
    is_early_stop = False

    start_time = datetime.datetime.now()
    for epoch_i in range(epoch):
        torch.cuda.empty_cache() # 清理无用的cuda中间变量缓存

        train_start_time = datetime.datetime.now()

        train_average_loss = train(model, optimizer, train_data_loader, loss, device)

        torch.save(model.state_dict(), save_param_dir + model_name + str(np.mod(epoch_i + 1, 5)) + '.pth')

        auc = test(model, valid_data_loader, device)
        valid_aucs.append(auc)

        train_end_time = datetime.datetime.now()
        print('epoch:', epoch_i, 'training average loss:', train_average_loss, 'validation: auc', auc,
              '[{}s]'.format((train_end_time - train_start_time).seconds))

        if eva_stopping(valid_aucs):
            early_stop_index = np.mod(epoch_i - 4, 5)
            is_early_stop = True
            break

    end_time = datetime.datetime.now()

    if is_early_stop:
        test_model = get_model(model_name, feature_nums, field_nums, latent_dims).to(device)
        load_path = save_param_dir + model_name + str(early_stop_index) + '.pth'
        test_model.load_state_dict(torch.load(load_path, map_location=device))  # 加载最优参数
    else:
        test_model = model

    auc = test(test_model, test_data_loader, device)
    torch.save(test_model.state_dict(), save_param_dir + model_name + 'best.pth') # 存储最优参数

    print('\ntest auc:', auc, datetime.datetime.now(), '[{}s]'.format((end_time - start_time).seconds))

    submission_path = data_path + dataset_name + campaign_id + model_name + '/' # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    days = day_indexs[:, 0]  # 数据集中有的日期

    day_aucs = []
    for day in days:
        current_day_index = day_indexs[days == day]
        data_index_start = current_day_index[0, 1]
        data_index_end = current_day_index[0, 2] + 1

        current_data = torch.tensor(train_fm[data_index_start: data_index_end, 1:]).to(device)
        y_labels = train_fm[data_index_start: data_index_end, 0]

        with torch.no_grad():
            y_pred = test_model(current_data).cpu().numpy()

            day_aucs.append([day, roc_auc_score(y_labels, y_pred.flatten())])

            y_pred_df = pd.DataFrame(data=y_pred)
            day_aucs_df = pd.DataFrame(data=day_aucs)

            y_pred_df.to_csv(submission_path + str(day) + '_test_submission.csv', header=None)
            day_aucs_df.to_csv(submission_path + 'day_aucs.csv', header=None)

def eva_stopping(valid_aucs): # early stopping
    if len(valid_aucs) > 5:
        if valid_aucs[-1] < valid_aucs[-2] and valid_aucs[-2] < valid_aucs[-3] and valid_aucs[-3] < valid_aucs[-4] and valid_aucs[-4] < valid_aucs[-5]:
            return True

    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--valid_day', default=12, help='6, 7, 8, 9, 10, 11, 12')
    parser.add_argument('--test_day', default=6, help='6, 7, 8, 9, 10, 11, 12')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3386')
    parser.add_argument('--model_name', default='LR', help='LR, FM, FFM')
    parser.add_argument('--latent_dims', default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_param_dir', default='models/model_params/')

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(1)

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
        args.device,
        args.save_param_dir
    )