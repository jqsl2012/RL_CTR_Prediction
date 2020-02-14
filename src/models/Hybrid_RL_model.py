import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

class Discrete_Actor(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(Discrete_Actor, self).__init__()
        self.input_dims = input_dims

        deep_input_dims = self.input_dims
        layers = list()
        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, action_nums))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        actions_value = self.mlp(input)

        return actions_value

class Continuous_Actor(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(Continuous_Actor, self).__init__()
        self.input_dims = input_dims

        self.bn_input = nn.BatchNorm1d(1)

        deep_input_dims = self.input_dims + 1
        layers = list()
        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, action_nums))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input, discrete_a):
        obs = torch.cat([input, self.bn_input(discrete_a)], dim=1)

        out = torch.softmax(self.mlp(obs), dim=1)

        return out


class Critic(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(Critic, self).__init__()

        self.bn_input = nn.BatchNorm1d(1)

        deep_input_dims = input_dims + action_nums + 1
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input, action, discrete_a):
        obs = torch.cat([input, self.bn_input(discrete_a)], dim=1)
        cat = torch.cat([obs, action], dim=1)

        q_out = self.mlp(cat)

        return q_out

class Hybrid_RL_Model():
    def __init__(
            self,
            feature_nums,
            field_nums=15,
            latent_dims=5,
            action_nums=2,
            campaign_id='1458',
            lr_A=5e-4,
            lr_C=1e-3,
            reward_decay=1,
            memory_size=4096000,
            batch_size=256,
            tau=0.005, # for target network soft update
            device='cuda:0',
    ):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.c_a_action_nums = action_nums
        self.d_actions_nums = action_nums - 1
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

        self.input_dims = self.field_nums * (self.field_nums - 1) // 2 + self.field_nums * self.latent_dims

        self.memory_state = torch.zeros(size=[self.memory_size, self.field_nums]).to(self.device)
        self.memory_action_reward = torch.zeros(size=[self.memory_size, self.c_a_action_nums + 1]).to(self.device)
        self.memory_discrete_action = torch.zeros(size=[self.memory_size, 1]).to(self.device)

        self.Continuous_Actor = Continuous_Actor(self.input_dims, self.c_a_action_nums).to(self.device)
        self.Discrete_Actor = Discrete_Actor(self.input_dims, self.d_actions_nums).to(self.device)
        self.Critic = Critic(self.input_dims, self.c_a_action_nums).to(self.device)
        
        self.Continuous_Actor_ = Continuous_Actor(self.input_dims, self.c_a_action_nums).to(self.device)
        self.Discrete_Actor_ = Discrete_Actor(self.input_dims, self.d_actions_nums).to(self.device)
        self.Critic_ = Critic(self.input_dims, self.c_a_action_nums).to(self.device)

        # 优化器
        self.optimizer_c_a = torch.optim.Adam(self.Continuous_Actor.parameters(), lr=self.lr_A, weight_decay=1e-5)
        self.optimizer_d_a = torch.optim.Adam(self.Discrete_Actor.parameters(), lr=self.lr_A, weight_decay=1e-5)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C, weight_decay=1e-5)

        self.loss_func = nn.MSELoss(reduction='mean')

    def store_transition(self, features, action_rewards, discrete_actions):
        transition_lens = len(features)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        index_end = (self.memory_counter + transition_lens) % self.memory_size

        if index_end > index_start:
            self.memory_state[index_start: index_end, :] = features  # 替换
            self.memory_action_reward[index_start: index_end, :] = action_rewards
            self.memory_discrete_action[index_start: index_end, :] = discrete_actions
        else:
            replace_len_1 = self.memory_size - index_start
            self.memory_state[index_start: self.memory_size, :] = features[0: replace_len_1]
            self.memory_action_reward[index_start: self.memory_size, :] = action_rewards[0: replace_len_1]
            self.memory_discrete_action[index_start: self.memory_size, :] = discrete_actions[0: replace_len_1]

            replace_len_2 = transition_lens - replace_len_1
            self.memory_state[0: replace_len_2, :] = features[replace_len_1: transition_lens]
            self.memory_action_reward[0: replace_len_2, :] = action_rewards[replace_len_1: transition_lens]
            self.memory_discrete_action[0: replace_len_2, :] = discrete_actions[replace_len_1: transition_lens]

        self.memory_counter += transition_lens

    def choose_continuous_action(self, state, discrete_a, exploration_rate):
        self.Continuous_Actor.eval()
        with torch.no_grad():
            action = self.Continuous_Actor.forward(state, discrete_a)
        self.Continuous_Actor.train()

        random_seeds = torch.rand(len(state), 1).to(self.device)

        random_action = torch.softmax(torch.normal(action, exploration_rate), dim=1)

        actions = torch.where(random_seeds >= exploration_rate, action,
                              random_action)

        return actions
    
    def choose_discrete_action(self, state, exploration_rate):
        self.Discrete_Actor.eval()
        with torch.no_grad():
            action_values = self.Discrete_Actor.forward(state)
        self.Discrete_Actor.train()

        random_seeds = torch.rand(len(state), 1).to(self.device)
        max_action = torch.argsort(-action_values)[:, 0] + 2
        random_action = torch.randint(low=2, high=self.d_actions_nums + 2, size=[len(state), 1]).to(self.device)

        actions = torch.where(random_seeds >= exploration_rate, max_action.view(-1, 1), random_action)

        return actions

    def choose_best_continuous_action(self, state, discrete_a):
        self.Continuous_Actor.eval()
        with torch.no_grad():
            action = self.Continuous_Actor.forward(state, discrete_a)

        return action
    
    def choose_best_discrete_action(self, state):
        self.Discrete_Actor.eval()
        with torch.no_grad():
            action_values = self.Discrete_Actor.forward(state)
            action = torch.argsort(-action_values)[:, 0] + 2

        return action.view(-1, 1)

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def sample_batch(self):
        if self.memory_counter > self.memory_size:
            sample_index = torch.LongTensor(random.sample(range(self.memory_size), self.batch_size)).to(self.device)
        else:
            sample_index = torch.LongTensor(random.sample(range(self.memory_counter), self.batch_size)).to(self.device)

        batch_memory_states = self.memory_state[sample_index, :].long()
        batch_memory_action_rewards = self.memory_action_reward[sample_index, :]
        b_discrete_a = self.memory_discrete_action[sample_index, :]

        b_s = batch_memory_states
        b_a = batch_memory_action_rewards[:, 0: self.c_a_action_nums]
        b_r = torch.unsqueeze(batch_memory_action_rewards[:, self.c_a_action_nums], 1)
        b_s_ = batch_memory_states

        return b_s, b_a, b_r, b_s_, b_discrete_a

    def learn_c(self, b_s, b_a, b_r, b_s_, b_discrete_a):
        q_target = b_r + self.gamma * self.Critic_.forward(b_s_, self.Continuous_Actor_.forward(b_s_, b_discrete_a),
                                                           b_discrete_a).detach()
        q = self.Critic.forward(b_s, b_a, b_discrete_a)

        td_error = self.loss_func(q, q_target)

        self.optimizer_c.zero_grad()
        td_error.backward()
        self.optimizer_c.step()

        td_error_r = td_error.item()

        return td_error_r

    def learn_c_a(self, b_s, b_discrete_a):

        c_a_loss = -self.Critic.forward(b_s, self.Continuous_Actor.forward(b_s, b_discrete_a), b_discrete_a).mean()

        self.optimizer_c_a.zero_grad()
        c_a_loss.backward()
        self.optimizer_c_a.step()

        return c_a_loss.item()

    def learn_d_a(self, b_s, b_discrete_a, b_r, b_s_):
        q_eval = self.Discrete_Actor.forward(b_s).gather(1, b_discrete_a.long() - 2)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next = self.Discrete_Actor_.forward(b_s_).detach()  # detach from graph, don't backpropagate，因为target网络不需要训练
        # 下一状态s的eval_net值
        q_eval_next = self.Discrete_Actor.forward(b_s_)
        max_b_a_next = torch.unsqueeze(torch.max(q_eval_next, 1)[1], 1)  # 选择最大Q的动作
        select_q_next = q_next.gather(1, max_b_a_next)

        q_target = b_r + self.gamma * select_q_next  # shape (batch, 1)

        # 训练eval_net
        d_a_loss = self.loss_func(q_eval, q_target)

        self.optimizer_d_a.zero_grad()
        d_a_loss.backward()
        self.optimizer_d_a.step()

        return d_a_loss.item()

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
