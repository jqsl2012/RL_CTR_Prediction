import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict

from src.models.Feature_embedding import Feature_Embedding

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

neural_nums_a_1 = 1024
neural_nums_a_1_action = 64
neural_nums_a_2 = 512
neural_nums_a_3 = 256
neural_nums_a_4 = 128
neural_nums_c_1_state = 1024
neural_nums_c_1_action = 64
neural_nums_a_1_pg_action = 64
neural_nums_c_2 = 512
neural_nums_c_3 = 256
neural_nums_c_4 = 128


class Actor(nn.Module):
    def __init__(self, input_dims, action_numbers, feature_nums, field_nums, latent_dims, campaign_id):
        super(Actor, self).__init__()
        # self.bn_pg = nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.embedding_layer = Feature_Embedding(feature_nums, field_nums, latent_dims, campaign_id)

        self.fc1 = nn.Linear(input_dims, neural_nums_a_1)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc_pg_a = nn.Linear(1, neural_nums_a_1_action)
        self.fc_pg_a.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(neural_nums_a_1 + neural_nums_a_1_action, neural_nums_a_2)
        self.fc2.weight.data.normal_(0, 0.1)

        self.fc3 = nn.Linear(neural_nums_a_2, neural_nums_a_3)
        self.fc3.weight.data.normal_(0, 0.1)

        self.fc4 = nn.Linear(neural_nums_a_3, neural_nums_a_4)
        self.fc4.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(neural_nums_a_4, action_numbers)
        self.out.weight.data.normal_(0, 0.1)

        self.dp = nn.Dropout(0.2)

    def forward(self, input, pg_a):
        input = self.embedding_layer.forward(input)

        input_ddpg_pg = torch.cat([F.relu(self.fc1(input)), F.relu(self.fc_pg_a(pg_a))], dim=1)

        x = self.dp(input_ddpg_pg)

        x_ = F.relu(self.fc2(x))
        x_ = self.dp(x_)

        x_1 = F.relu(self.fc3(x_))
        x_1 = self.dp(x_1)

        x_2 = F.relu(self.fc4(x_1))
        x_2 = self.dp(x_2)

        out = torch.softmax(self.out(x_2), dim=1)

        return out


class Critic(nn.Module):
    def __init__(self, input_dims, action_numbers, feature_nums, field_nums, latent_dims, campaign_id):
        super(Critic, self).__init__()
        # self.bn_pg = nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.embedding_layer = Feature_Embedding(feature_nums, field_nums, latent_dims, campaign_id)

        self.fc_s = nn.Linear(input_dims, neural_nums_c_1_state)
        self.fc_s.weight.data.normal_(0, 0.1)

        self.fc_a = nn.Linear(action_numbers, neural_nums_c_1_action * action_numbers)
        self.fc_a.weight.data.normal_(0, 0.1)

        self.fc_pg_a = nn.Linear(1, neural_nums_a_1_pg_action)
        self.fc_pg_a.weight.data.normal_(0, 0.1)

        self.fc_q = nn.Linear(neural_nums_c_1_state + neural_nums_c_1_action * action_numbers + neural_nums_a_1_pg_action, neural_nums_c_2)
        self.fc_q.weight.data.normal_(0, 0.1)

        self.fc_ = nn.Linear(neural_nums_c_2, neural_nums_c_3)
        self.fc_.weight.data.normal_(0, 0.1)

        self.fc_1 = nn.Linear(neural_nums_c_3, neural_nums_c_4)
        self.fc_1.weight.data.normal_(0, 0.1)

        self.fc_out = nn.Linear(neural_nums_c_4, 1)
        self.fc_out.weight.data.normal_(0, 0.1)

        self.dp = nn.Dropout(0.2)

    def forward(self, input, action, pg_a):
        input = self.embedding_layer.forward(input)

        f_s = F.relu(self.fc_s(input))
        f_a = F.relu(self.fc_a(action))
        f_pg_a = F.relu(self.fc_pg_a(pg_a))

        cat = torch.cat([f_s, f_a, f_pg_a], dim=1)
        cat = self.dp(cat)

        q_ = F.relu(self.fc_q(cat))
        q_ = self.dp(q_)

        q_1 = F.relu(self.fc_(q_))
        q_1 = self.dp(q_1)

        q_2 = F.relu(self.fc_1(q_1))
        q_2 = self.dp(q_2)

        q_out = self.fc_out(q_2)

        return q_out

class DDPG():
    def __init__(
            self,
            feature_nums,
            field_nums=15,
            latent_dims=5,
            action_nums=2,
            campaign_id='1458',
            lr_A=1e-4,
            lr_C=1e-3,
            reward_decay=1,
            memory_size=4096000,
            batch_size=256,
            tau=0.001, # for target network soft update
            device='cuda:0',
    ):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.action_nums = action_nums
        self.campaign_id = campaign_id
        self.lr_A = lr_A
        self.lr_C = lr_C
        self.gamma = reward_decay
        self.latent_dims = latent_dims
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.device = device

        self.memory_counter = 0

        input_dims = 0
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                input_dims += self.latent_dims
        input_dims += self.latent_dims * self.field_nums  # 15+75
        self.input_dims = input_dims

        self.memory_state = torch.zeros(size=[self.memory_size, self.field_nums]).to(self.device)
        self.memory_action_reward = torch.zeros(size=[self.memory_size, self.action_nums + 1]).to(self.device)
        self.memory_pg_action = torch.zeros(size=[self.memory_size, 1]).to(self.device)

        self.Actor = Actor(self.input_dims, self.action_nums, self.feature_nums, self.field_nums, self.latent_dims, self.campaign_id).to(self.device)
        self.Critic = Critic(self.input_dims, self.action_nums, self.feature_nums, self.field_nums, self.latent_dims, self.campaign_id).to(self.device)

        self.Actor_ = Actor(self.input_dims, self.action_nums, self.feature_nums, self.field_nums, self.latent_dims, self.campaign_id).to(self.device)
        self.Critic_ = Critic(self.input_dims, self.action_nums, self.feature_nums, self.field_nums, self.latent_dims, self.campaign_id).to(self.device)

        # 优化器
        self.optimizer_a = torch.optim.Adam(self.Actor.parameters(), lr=self.lr_A, weight_decay=1e-5)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C, weight_decay=1e-5)

        self.loss_func = nn.MSELoss(reduction='mean')

    # def load_embedding(self, pretrain_params):
    #     for i, embedding in enumerate(self.embedding_layer.field_feature_embeddings):
    #         embedding.weight.data.copy_(
    #             torch.from_numpy(
    #                 np.array(pretrain_params['field_feature_embeddings.' + str(i) + '.weight'].cpu())
    #             )
    #         )

    # def load_embedding(self, pretrain_params):
    #     self.embedding_layer.feature_embedding.weight.data.copy_(
    #         torch.from_numpy(
    #             np.array(pretrain_params['feature_embedding.weight'].cpu())
    #         )
    #     )

    def store_transition(self, features, action_rewards, pg_actions):
        transition_lens = len(features)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        index_end = (self.memory_counter + transition_lens) % self.memory_size

        if index_end > index_start:
            self.memory_state[index_start: index_end, :] = features  # 替换
            self.memory_action_reward[index_start: index_end, :] = action_rewards
            self.memory_pg_action[index_start: index_end, :] = pg_actions
        else:
            replace_len_1 = self.memory_size - index_start
            self.memory_state[index_start: self.memory_size, :] = features[0: replace_len_1]
            self.memory_action_reward[index_start: self.memory_size, :] = action_rewards[0: replace_len_1]
            self.memory_pg_action[index_start: self.memory_size, :] = pg_actions[0: replace_len_1]

            replace_len_2 = transition_lens - replace_len_1
            self.memory_state[0: replace_len_2, :] = features[replace_len_1: transition_lens]
            self.memory_action_reward[0: replace_len_2, :] = action_rewards[replace_len_1: transition_lens]
            self.memory_pg_action[0: replace_len_2, :] = pg_actions[replace_len_1: transition_lens]

        self.memory_counter += transition_lens

    def choose_action(self, state, pg_a):

        # state = self.embedding_layer.forward(state)

        self.Actor.eval()
        with torch.no_grad():
            action = self.Actor.forward(state, pg_a)
        self.Actor.train()

        return action

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def sample_batch(self):
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory_states = self.memory_state[sample_index, :].long()
        batch_memory_action_rewards = self.memory_action_reward[sample_index, :]
        batch_memory_pg_actions = self.memory_pg_action[sample_index, :]

        # b_s = self.embedding_layer.forward(batch_memory_states)
        b_s = batch_memory_states
        b_a = batch_memory_action_rewards[:, 0: self.action_nums]
        b_r = torch.unsqueeze(batch_memory_action_rewards[:, self.action_nums], 1)
        # b_s_ = self.embedding_layer.forward(batch_memory_states)
        b_s_ = batch_memory_states

        return b_s, b_a, b_r, b_s_, batch_memory_pg_actions

    def learn_c(self, b_s, b_a, b_r, b_s_, b_pg_a):
        q_target = b_r + self.gamma * self.Critic_.forward(b_s_, self.Actor_.forward(b_s_, b_pg_a), b_pg_a)
        q = self.Critic.forward(b_s, b_a, b_pg_a)
        td_error = F.smooth_l1_loss(q, q_target.detach())

        self.optimizer_c.zero_grad()
        td_error.backward()
        self.optimizer_c.step()

        td_error_r = td_error.item()

        return td_error_r

    def learn_a(self, b_s, b_pg_a):
        a_loss = -self.Critic.forward(b_s, self.Actor.forward(b_s, b_pg_a), b_pg_a).mean()

        self.optimizer_a.zero_grad()
        a_loss.backward()
        self.optimizer_a.step()

        a_loss_r = a_loss.item()

        torch.cuda.empty_cache()

        return a_loss_r

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.01, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
