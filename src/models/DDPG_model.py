import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

neural_nums_a_1 = 1024
neural_nums_a_2 = 512
neural_nums_a_3 = 256
neural_nums_a_4 = 128
neural_nums_c_1_state = 1024
neural_nums_c_1_action = 64
neural_nums_c_2 = 512
neural_nums_c_3 = 256
neural_nums_c_4 = 128

class Fature_embedding(nn.Module):
    def __init__(self, feature_numbers, field_nums, latent_dims, campaign_id):
        super(Fature_embedding, self).__init__()
        self.field_nums = field_nums
        self.latent_dims = latent_dims
        self.campaign_id = campaign_id

        self.pretrain_params = torch.load('models/model_params/' + self.campaign_id + '/FFMbest.pth')

        self.field_feature_embeddings = nn.ModuleList([
            nn.Embedding(feature_numbers, latent_dims) for _ in range(field_nums)
        ])  # 相当于建立一个field_nums * feature_nums * latent_dims的三维矩阵
        for i, embedding in enumerate(self.field_feature_embeddings):
            self.field_feature_embeddings[i].weight.data.copy_(
                torch.from_numpy(
                    np.array(self.pretrain_params['field_feature_embeddings.' + str(i) + '.weight'].cpu())
                )
            )

        self.linear = nn.Embedding(feature_numbers, 1)
        self.linear.weight.data.copy_(
            torch.from_numpy(
                np.array(self.pretrain_params['linear.weight'].cpu())
            )
        )

    def forward(self, x):
        x_second_embedding = [self.field_feature_embeddings[i](x) for i in range(self.field_nums)]
        embedding_vectors = torch.FloatTensor().cuda()
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                hadamard_product = x_second_embedding[j][:, i] * x_second_embedding[i][:, j]
                inner_product = torch.sum(hadamard_product, dim=1).view(-1, 1).detach()
                embedding_vectors = torch.cat([embedding_vectors, hadamard_product, inner_product], dim=1)

        for i, embedding in enumerate(self.field_feature_embeddings):
            embedding_vectors = torch.cat([embedding_vectors, embedding(x[:, i])], dim=1)

        x_linear_embedding = self.linear(x).view(-1, self.field_nums)

        embedding_vectors = torch.cat([embedding_vectors, x_linear_embedding], dim=1)

        return embedding_vectors.detach()
        # return self.field_feature_embeddings(x).view(-1, self.field_nums * self.latent_dims) # m * n矩阵平铺为1 * [m*n]

class Actor(nn.Module):
    def __init__(self, input_dims, action_numbers):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, neural_nums_a_1)
        self.fc2 = nn.Linear(neural_nums_a_1, neural_nums_a_2)
        self.fc3 = nn.Linear(neural_nums_a_2, neural_nums_a_3)
        self.fc4 = nn.Linear(neural_nums_a_3, neural_nums_a_4)
        self.out = nn.Linear(neural_nums_a_4, action_numbers)

        self.dp = nn.Dropout(0.5)


    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.dp(x)

        x_ = F.relu(self.fc2(x))
        x_ = self.dp(x_)

        x_1 = F.relu(self.fc3(x_))
        x_1 = self.dp(x_1)

        x_2 = F.relu(self.fc4(x_1))
        x_2 = self.dp(x_2)

        out = torch.sigmoid(self.out(x_2))

        return out

class Critic(nn.Module):
    def __init__(self, input_dims, action_numbers):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(input_dims, neural_nums_c_1_state)
        self.fc_a = nn.Linear(action_numbers, neural_nums_c_1_action)
        self.fc_q = nn.Linear(neural_nums_c_1_state + neural_nums_c_1_action, neural_nums_c_2)
        self.fc_ = nn.Linear(neural_nums_c_2, neural_nums_c_3)
        self.fc_1 = nn.Linear(neural_nums_c_3, neural_nums_c_4)
        self.fc_out = nn.Linear(neural_nums_c_4, 1)

        self.dp = nn.Dropout(0.5)

    def forward(self, input, action):
        f_s = F.relu(self.fc_s(input))
        f_a = self.fc_a(action)
        cat = torch.cat([f_s, f_a], dim=1)
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
            action_nums=1,
            latent_dims=5,
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
                input_dims += 6
        input_dims += 90  # 15+75
        self.input_dims = input_dims

        self.memory_state = np.zeros((self.memory_size, self.field_nums))
        self.memory_action_reward = np.zeros((self.memory_size, self.action_nums + 1))

        self.embedding_layer = Fature_embedding(self.feature_nums, self.field_nums, self.latent_dims, self.campaign_id).to(self.device)

        self.Actor = Actor(self.input_dims, self.action_nums).to(self.device)
        self.Critic = Critic(self.input_dims, self.action_nums).to(self.device)

        self.Actor_ = Actor(self.input_dims, self.action_nums).to(self.device)
        self.Critic_ = Critic(self.input_dims, self.action_nums).to(self.device)

        # 优化器
        self.optimizer_a = torch.optim.Adam(self.Actor.parameters(), lr=self.lr_A, weight_decay=1e-5)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C, weight_decay=1e-5)

        self.loss_func = nn.MSELoss(reduction='mean')

    def store_transition(self, features, action_rewards):

        features = features.cpu().detach().numpy()
        action_rewards = action_rewards.cpu().detach().numpy()
        transition_lens = len(features)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        index_end = (self.memory_counter + transition_lens) % self.memory_size

        if index_end > index_start:
            self.memory_state[index_start: index_end, :] = features  # 替换
            self.memory_action_reward[index_start: index_end, :] = action_rewards
        else:
            replace_len_1 = self.memory_size - index_start
            self.memory_state[index_start: self.memory_size, :] = features[0: replace_len_1]
            self.memory_action_reward[index_start: self.memory_size, :] = action_rewards[0: replace_len_1]
            replace_len_2 = transition_lens - replace_len_1
            self.memory_state[0: replace_len_2, :] = features[replace_len_1: transition_lens]
            self.memory_action_reward[0: replace_len_2, :] = action_rewards[replace_len_1: transition_lens]

        self.memory_counter += transition_lens

    def choose_action(self, state):

        state = self.embedding_layer.forward(state)

        self.Actor.eval()
        with torch.no_grad():
            action = self.Actor.forward(state).cpu().numpy()
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

        batch_memory_state = self.memory_state[sample_index, :]
        batch_memory_action_rewards = self.memory_action_reward[sample_index, :]

        b_s = self.embedding_layer.forward(torch.LongTensor(batch_memory_state).to(self.device))
        b_a = torch.FloatTensor(batch_memory_action_rewards[:, 0: self.action_nums]).to(self.device)
        b_r = torch.unsqueeze(torch.FloatTensor(batch_memory_action_rewards[:, self.action_nums]).to(self.device), 1)
        b_s_ = self.embedding_layer.forward(torch.LongTensor(batch_memory_state).to(self.device))

        return b_s, b_a, b_r, b_s_

    def learn_c(self, b_s, b_a, b_r, b_s_):
        q_target = b_r + self.gamma * self.Critic_.forward(b_s_, self.Actor_.forward(b_s_))
        q = self.Critic.forward(b_s, b_a)
        td_error = F.smooth_l1_loss(q, q_target.detach())
        self.optimizer_c.zero_grad()
        td_error.backward()
        self.optimizer_c.step()

        td_error_r = td_error.item()

        return td_error_r

    def learn_a(self, b_s):
        a_loss = -self.Critic.forward(b_s, self.Actor.forward(b_s)).mean()

        self.optimizer_a.zero_grad()
        a_loss.backward()
        self.optimizer_a.step()

        a_loss_r = a_loss.item()

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
