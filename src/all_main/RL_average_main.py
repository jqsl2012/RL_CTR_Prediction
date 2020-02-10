import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
# import src.models.PG_model as Model
import src.models.p_model as p_model
import src.models.DDPG_for_avg_model as DDPG_for_avg_model
import src.models.creat_data as Data

import torch
import torch.nn as nn
import torch.utils.data

np.seterr(all='raise')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(action_nums, feature_nums, field_nums, latent_dims, batch_size, memory_size, device, campaign_id):
    ddpg_for_avg_Model = DDPG_for_avg_model.DDPG(feature_nums, field_nums, latent_dims,
                                               action_nums=action_nums,
                                               campaign_id=campaign_id, batch_size=batch_size,
                                               memory_size=memory_size, device=device)
    return ddpg_for_avg_Model


def get_dataset(datapath, dataset_name, campaign_id):
    data_path = datapath + dataset_name + campaign_id
    train_data_file_name = 'train_.txt'
    train_fm = pd.read_csv(data_path + train_data_file_name, header=None).values.astype(int)

    test_data_file_name = 'test_.txt'
    test_fm = pd.read_csv(data_path + test_data_file_name, header=None).values.astype(int)

    field_nums = len(train_fm[0, 1:])  # 特征域的数量

    feature_index_name = 'featindex.txt'
    feature_index = pd.read_csv(data_path + feature_index_name, header=None).values
    feature_nums = int(feature_index[-1, 0].split('\t')[1]) + 1  # 特征数量

    train_data = train_fm
    test_data = test_fm

    return train_fm, train_data, test_data, field_nums, feature_nums

def generate_preds(model_dict, features, prob_weights, device):
    basic_rewards = torch.ones(size=[len(features), 1]).to(device)

    pretrain_model_len = len(model_dict)  # 有多少个预训练模型

    pretrain_y_preds = {}
    for i in range(pretrain_model_len):
        pretrain_y_preds[i] = model_dict[i](features).detach()

    current_pretrain_y_preds = torch.cat([
        pretrain_y_preds[l] for l in range(pretrain_model_len)
    ], dim=1)

    current_model_y_preds_avg = torch.mul(prob_weights, current_pretrain_y_preds).mean().view(-1, 1)
    current_pretrain_y_preds_avg = current_pretrain_y_preds.mean().view(-1, 1)

    rewards = torch.where(current_model_y_preds_avg >= current_pretrain_y_preds_avg,
                          basic_rewards * 1,
                          basic_rewards * -1)

    return current_model_y_preds_avg, rewards


def train(ddpg_for_avg_Model, model_dict, data_loader, exploration_rate, device):
    total_loss = 0
    log_intervals = 0
    total_rewards = 0
    targets, predicts = list(), list()

    for i, (features, labels) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)

        prob_actions, prob_weights = ddpg_for_avg_Model.choose_action(features, exploration_rate)

        y_preds, rewards = \
            generate_preds(model_dict, features, prob_weights, device)

        targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
        predicts.extend(y_preds.tolist())

        action_rewards = torch.cat([prob_weights, rewards], dim=1)
        ddpg_for_avg_Model.store_transition(features, action_rewards)

        b_s, b_a, b_r, b_s_ = ddpg_for_avg_Model.sample_batch()
        td_error = ddpg_for_avg_Model.learn_c(b_s, b_a, b_r, b_s_)
        a_loss = ddpg_for_avg_Model.learn_a(b_s)
        ddpg_for_avg_Model.soft_update(ddpg_for_avg_Model.Actor, ddpg_for_avg_Model.Actor_)
        ddpg_for_avg_Model.soft_update(ddpg_for_avg_Model.Critic, ddpg_for_avg_Model.Critic_)

        total_loss += td_error  # 取张量tensor里的标量值，如果直接返回train_loss很可能会造成GPU out of memory
        log_intervals += 1

        total_rewards += torch.sum(rewards, dim=0).item()

        torch.cuda.empty_cache()  # 清除缓存

    return total_loss / log_intervals, total_rewards / log_intervals, roc_auc_score(targets, predicts)


def test(ddpg_for_avg_Model, model_dict, data_loader, loss, device):
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    with torch.no_grad():
        for i, (features, labels) in enumerate(data_loader):
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            prob_actions, prob_weights = ddpg_for_avg_Model.choose_best_action(features)

            y, rewards = generate_preds(model_dict, features, prob_weights, device)

            test_loss = loss(y, labels.float())
            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

    return roc_auc_score(targets, predicts), total_test_loss / intervals


def submission(ddpg_for_avg_Model, model_dict, data_loader, device):
    targets, predicts = list(), list()
    final_prob_weights = torch.FloatTensor().to(device)
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            prob_actions, prob_weights = ddpg_for_avg_Model.choose_best_action(features)
            y, rewards = generate_preds(model_dict, features, prob_weights, device)

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())

            final_prob_weights = torch.cat([final_prob_weights, prob_weights], dim=0)

    return predicts, roc_auc_score(targets, predicts), final_prob_weights.cpu().numpy()


def main(data_path, dataset_name, campaign_id, latent_dims, model_name, epoch, batch_size, device, save_param_dir):
    if not os.path.exists(save_param_dir):
        os.mkdir(save_param_dir)

    device = torch.device(device)  # 指定运行设备
    train_fm, train_data, test_data, field_nums, feature_nums = get_dataset(data_path, dataset_name, campaign_id)

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    num_workers=8)  # 0.7153541503790021
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    FFM = p_model.FFM(feature_nums, field_nums, latent_dims)
    FFM_pretain_params = torch.load(save_param_dir + campaign_id + 'FFMbest.pth')
    FFM.load_state_dict(FFM_pretain_params)
    FFM.eval()

    FM = p_model.FM(feature_nums, latent_dims)
    FM_pretain_params = torch.load(save_param_dir + campaign_id + 'FMbest.pth')
    FM.load_state_dict(FM_pretain_params)
    FM.eval()

    AFM = p_model.AFM(feature_nums, field_nums, latent_dims)
    AFM_pretain_params = torch.load(save_param_dir + campaign_id + 'AFMbest.pth')
    AFM.load_state_dict(AFM_pretain_params)
    AFM.eval()

    WandD = p_model.WideAndDeep(feature_nums, field_nums, latent_dims)
    WandD_pretrain_params = torch.load(save_param_dir + campaign_id + 'W&Dbest.pth')
    WandD.load_state_dict(WandD_pretrain_params)
    WandD.eval()

    DeepFM = p_model.DeepFM(feature_nums, field_nums, latent_dims)
    DeepFM_pretrain_params = torch.load(save_param_dir + campaign_id + 'DeepFMbest.pth')
    DeepFM.load_state_dict(DeepFM_pretrain_params)
    DeepFM.eval()

    FNN = p_model.FNN(feature_nums, field_nums, latent_dims)
    FNN_pretrain_params = torch.load(save_param_dir + campaign_id + 'FNNbest.pth')
    FNN.load_state_dict(FNN_pretrain_params)
    FNN.eval()

    IPNN = p_model.InnerPNN(feature_nums, field_nums, latent_dims)
    IPNN_pretrain_params = torch.load(save_param_dir + campaign_id + 'IPNNbest.pth')
    IPNN.load_state_dict(IPNN_pretrain_params)
    IPNN.eval()

    OPNN = p_model.OuterPNN(feature_nums, field_nums, latent_dims)
    OPNN_pretrain_params = torch.load(save_param_dir + campaign_id + 'OPNNbest.pth')
    OPNN.load_state_dict(OPNN_pretrain_params)
    OPNN.eval()

    DCN = p_model.DCN(feature_nums, field_nums, latent_dims)
    DCN_pretrain_params = torch.load(save_param_dir + campaign_id + 'DCNbest.pth')
    DCN.load_state_dict(DCN_pretrain_params)
    DCN.eval()

    model_dict = {0: FM.to(device), 1: FFM.to(device), 2: WandD.to(device), 3: IPNN.to(device), 4: DCN.to(device)}

    model_dict_len = len(model_dict)

    memory_size = 1000000
    ddpg_for_avg_Model = get_model(model_dict_len, feature_nums, field_nums, latent_dims, batch_size,
                                              memory_size, device, campaign_id)

    loss = nn.BCELoss()

    valid_aucs = []
    valid_losses = []
    early_stop_index = 0
    is_early_stop = False

    start_time = datetime.datetime.now()
    exploration_rate = 1
    for epoch_i in range(epoch):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_start_time = datetime.datetime.now()

        train_average_loss, train_average_rewards, train_auc = train(ddpg_for_avg_Model, model_dict,
                                                                     train_data_loader, exploration_rate,
                                                                     device)

        auc, valid_loss = test(ddpg_for_avg_Model, model_dict, test_data_loader, loss, device)
        valid_aucs.append(auc)
        valid_losses.append(valid_loss)

        train_end_time = datetime.datetime.now()
        print('epoch:', epoch_i, 'training average loss:', train_average_loss, 'training average rewards',
              train_average_rewards, 'training auc', train_auc, 'validation auc:', auc,
              'validation loss:', valid_loss, '[{}s]'.format((train_end_time - train_start_time).seconds))

        exploration_rate *= 0.95

    end_time = datetime.datetime.now()

    auc, test_loss = test(ddpg_for_avg_Model, model_dict, test_data_loader, loss, device)
    print('\ntest auc:', auc, datetime.datetime.now(), '[{}s]'.format((end_time - start_time).seconds))

    submission_path = data_path + dataset_name + campaign_id + model_name + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    # 测试集submission
    test_predicts, test_auc, prob_weights = submission(ddpg_for_avg_Model, model_dict,
                                                                test_data_loader, device)

    prob_weights_df = pd.DataFrame(data=prob_weights)
    prob_weights_df.to_csv(submission_path + 'test_prob_weights.csv', header=None)

    test_pred_df = pd.DataFrame(data=test_predicts)

    test_pred_df.to_csv(submission_path + 'test_submission.csv', header=None)

    day_aucs = [[test_auc]]
    day_aucs_df = pd.DataFrame(data=day_aucs)
    day_aucs_df.to_csv(submission_path + 'day_aucs.csv', header=None)

    torch.save(ddpg_for_avg_Model.Actor.state_dict(),
               save_param_dir + campaign_id + '/ddpg_for_avg_Model' + 'best.pth')  # 存储最优参数


def eva_stopping(valid_aucs, valid_losses, type):  # early stopping
    if type == 'auc':
        if len(valid_aucs) > 5:
            if valid_aucs[-1] < valid_aucs[-2] and valid_aucs[-2] < valid_aucs[-3] and valid_aucs[-3] < valid_aucs[
                -4] and valid_aucs[-4] < valid_aucs[-5]:
                return True
    else:
        if len(valid_losses) > 5:
            if valid_losses[-1] > valid_losses[-2] and valid_losses[-2] > valid_losses[-3] and valid_losses[-3] > \
                    valid_losses[-4] and valid_losses[-4] > valid_losses[-5]:
                return True

    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3386')
    parser.add_argument('--model_name', default='PG_DDPG', help='LR, FM, FFM, W&D')
    parser.add_argument('--latent_dims', default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_type', default='auc', help='auc, loss')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_param_dir', default='../models/model_params/')

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(1)

    main(
        args.data_path,
        args.dataset_name,
        args.campaign_id,
        args.latent_dims,
        args.model_name,
        args.epoch,
        args.batch_size,
        args.device,
        args.save_param_dir
    )