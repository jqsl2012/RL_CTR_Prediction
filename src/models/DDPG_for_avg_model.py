import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import MultivariateNormal, Categorical
import datetime
from torch.distributions import Normal, Categorical, MultivariateNormal

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1)

class Memory(object):
    def __init__(self, memory_size, transition_lens, device):
        self.device = device
        self.transition_lens = transition_lens # 存储的数据长度
        self.epsilon = 1e-3 # 防止出现zero priority
        self.alpha = 0.6 # 取值范围(0,1)，表示td error对priority的影响
        self.beta = 0.4 # important sample， 从初始值到1
        self.beta_increment_per_sampling = 0.0001
        self.abs_err_upper = 1 # abs_err_upper和epsilon ，表明p优先级值的范围在[epsilon,abs_err_upper]之间，可以控制也可以不控制

        self.memory_size = memory_size
        self.memory_counter = 0

        self.prioritys_ = torch.zeros(size=[memory_size, 1]).to(self.device)
        # indexs = torch.range(0, self.memory_size)
        # self.prioritys_[:, 1] = indexs

        self.memory = torch.zeros(size=[memory_size, transition_lens]).to(self.device)

    def get_priority(self, td_error):
        return torch.pow(torch.abs(td_error) + self.epsilon, self.alpha)

    def add(self, td_error, transitions): # td_error是tensor矩阵
        transition_lens = len(transitions)
        p = self.get_priority(td_error)

        memory_start = self.memory_counter % self.memory_size
        memory_end = (self.memory_counter + len(transitions)) % self.memory_size

        if memory_end > memory_start:
            self.memory[memory_start: memory_end, :] = transitions
            self.prioritys_[memory_start: memory_end, :] = p
        else:
            replace_len_1 = self.memory_size - memory_start
            self.memory[memory_start: self.memory_size, :] = transitions[0: replace_len_1]
            self.prioritys_[memory_start: self.memory_size, :] = p[0: replace_len_1]

            replace_len_2 = transition_lens - replace_len_1
            self.memory[:replace_len_2, :] = transitions[replace_len_1: transition_lens]
            self.prioritys_[:replace_len_2, :] = p[replace_len_1: transition_lens]

        self.memory_counter += len(transitions)

    def stochastic_sample(self, batch_size):
        total_p = torch.sum(self.prioritys_, dim=0)

        if self.memory_counter >= self.memory_size:
            min_prob = torch.min(self.prioritys_)
            # 采样概率分布
            P = torch.div(self.prioritys_, total_p).squeeze(1).cpu().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_size, batch_size, p=P)).long().to(self.device)
        else:
            min_prob = torch.min(self.prioritys_[:self.memory_counter, :])
            P = torch.div(self.prioritys_[:self.memory_counter, :], total_p).squeeze(1).cpu().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_counter, batch_size, p=P)).long().to(self.device)

        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()

        batch = self.memory[sample_indexs]
        choose_priorities = self.prioritys_[sample_indexs]
        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta)

        return sample_indexs, batch, ISweights

    def greedy_sample(self, batch_size):
        # total_p = torch.sum(self.prioritys_, dim=0)

        if self.memory_counter >= self.memory_size:
            min_prob = torch.min(self.prioritys_)
        else:
            min_prob = torch.min(self.prioritys_[:self.memory_counter, :])
        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()

        sorted_priorities, sorted_indexs = torch.sort(-self.prioritys_, dim=0)

        choose_idxs = sorted_indexs[:batch_size, :].squeeze(1)

        batch = self.memory[choose_idxs]

        choose_priorities = -sorted_priorities[:batch_size, :]

        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta).detach()

        return choose_idxs, batch, ISweights

    def batch_update(self, choose_idx, td_errors):
        p = self.get_priority(td_errors)
        self.prioritys_[choose_idx, :] = p

class Critic(nn.Module):
    def __init__(self, input_dims, c_action_nums):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.c_action_nums = c_action_nums

        self.bn_input = nn.BatchNorm1d(self.input_dims)

        deep_input_dims = self.input_dims + self.c_action_nums

        neuron_nums = [300, 300, 300]
        self.mlp = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
            nn.Linear(neuron_nums[2], 1)
        )

    def evaluate(self, input, c_actions): # actions 包括连续与非连续
        obs = self.bn_input(input)
        cat = torch.cat([obs, c_actions], dim=1)

        q_out = self.mlp(cat)

        return q_out

class hybrid_actors(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(hybrid_actors, self).__init__()
        self.input_dims = input_dims
        self.c_action_dims = action_nums
        self.d_action_dims = action_nums - 1

        self.bn_input = nn.BatchNorm1d(self.input_dims)

        neuron_nums = [300, 300, 300]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
            nn.Linear(neuron_nums[2], self.c_action_dims),
            nn.Softmax(dim=-1)
        )

        self.c_action_std = nn.Parameter(torch.zeros(size=[1]))

    def act(self, input):
        obs = self.bn_input(input)
        c_action_means = self.mlp(obs)

        c_action_dist = Normal(c_action_means, F.softplus(self.c_action_std))

        ensemble_c_actions = torch.softmax(c_action_dist.sample(), dim=-1)

        return ensemble_c_actions

    def evaluate(self, input):
        obs = self.bn_input(input)
        c_actions_means = self.mlp(obs)

        c_action_dist = Normal(c_actions_means, F.softplus(self.c_action_std))
        c_action_entropy = c_action_dist.entropy()

        return c_actions_means, c_action_entropy

class DDPG_AVG():
    def __init__(
            self,
            feature_nums,
            field_nums=15,
            latent_dims=5,
            action_nums=2,
            campaign_id='1458',
            lr_C_A=1e-3,
            lr_D_A=1e-3,
            lr_C=1e-2,
            reward_decay=1,
            memory_size=4096000,
            batch_size=256,
            tau=0.005,  # for target network soft update
            device='cuda:0',
    ):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.c_action_nums = action_nums
        self.d_action_nums = action_nums - 1
        self.campaign_id = campaign_id
        self.lr_C_A = lr_C_A
        self.lr_D_A = lr_D_A
        self.lr_C = lr_C
        self.gamma = reward_decay
        self.latent_dims = latent_dims
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.device = device

        self.memory_counter = 0

        self.input_dims = self.field_nums * (self.field_nums - 1) // 2 + self.field_nums * self.latent_dims

        self.memory = Memory(self.memory_size, self.field_nums + self.c_action_nums + 1, self.device)

        self.Hybrid_Actor = hybrid_actors(self.input_dims, self.c_action_nums).to(self.device)
        self.Critic = Critic(self.input_dims, self.c_action_nums).to(self.device)

        self.Hybrid_Actor_ = hybrid_actors(self.input_dims, self.c_action_nums).to(self.device)
        self.Critic_ = Critic(self.input_dims, self.c_action_nums).to(self.device)

        # 优化器
        self.optimizer_a = torch.optim.Adam(self.Hybrid_Actor.parameters(), lr=self.lr_C_A, weight_decay=1e-5)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C, weight_decay=1e-5)

        self.loss_func = nn.MSELoss(reduction='mean')

        self.c_action_std = torch.ones(size=[1, self.c_action_nums]).to(self.device)

    def store_transition(self, transitions, embedding_layer): # 所有的值都应该弄成float
        b_s = embedding_layer.forward(transitions[:, :self.field_nums].long())
        b_s_ = b_s
        b_c_a = transitions[:, self.field_nums: self.field_nums + self.c_action_nums]
        b_r = torch.unsqueeze(transitions[:, -1], dim=1)

        # current state's action_values
        c_actions_means, c_actions_entropy = self.Hybrid_Actor.evaluate(b_s)

        # critic
        q_target_critic = b_r + self.gamma * self.Critic_.evaluate(b_s_, c_actions_means)
        q_critic = self.Critic.evaluate(b_s, b_c_a)
        td_error_critic = q_target_critic - q_critic

        td_errors = td_error_critic

        self.memory.add(td_errors.detach(), transitions)

    def choose_action(self, state):
        self.Hybrid_Actor.eval()
        with torch.no_grad():
            c_actions = self.Hybrid_Actor.act(state)

        self.Hybrid_Actor.train()
        return c_actions

    def choose_best_action(self, state):
        self.Hybrid_Actor.eval()
        with torch.no_grad():
            c_action_means, c_entropy = self.Hybrid_Actor.evaluate(state)

        ensemble_c_actions = c_action_means

        return ensemble_c_actions

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, embedding_layer):
        # sample
        choose_idx, batch_memory, ISweights = self.memory.stochastic_sample(self.batch_size)

        b_s = embedding_layer.forward(batch_memory[:, :self.field_nums].long())
        b_c_a = batch_memory[:, self.field_nums: self.field_nums + self.c_action_nums]
        b_r = torch.unsqueeze(batch_memory[:, -1], 1)
        b_s_ = b_s  # embedding_layer.forward(batch_memory_states)

        # Critic
        c_actions_means_for_critic, c_actions_entropy_for_critic = self.Hybrid_Actor.evaluate(b_s)
        q_target = b_r + self.gamma * self.Critic_.evaluate(b_s_, c_actions_means_for_critic)
        q = self.Critic.evaluate(b_s, b_c_a)

        critic_td_error = (q_target - q).detach()
        critic_loss = (ISweights * torch.pow(q - q_target.detach(), 2)).mean()

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        self.optimizer_c.step()

        critic_loss_r = critic_loss.item()

        # current state's action_values
        c_actions_means, c_actions_entropy = self.Hybrid_Actor.evaluate(b_s)
        # next state's action_values
        c_actions_means_, c_actions_entropy_ = self.Hybrid_Actor.evaluate(b_s)

        # Hybrid_Actor
        # c a
        c_a_loss = -self.Critic.evaluate(b_s, c_actions_means).mean()
        # print(c_actions_entropy.mean(), d_actions_entropy.mean())
        # d a
        # q_eval = d_actions_q_values.gather(1, b_discrete_a.long() - 2)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        # q_next = d_actions_q_values_
        # q_target = b_r + self.gamma * q_next.max(1)[0].view(-1, 1)  # shape (batch, 1)
        # # d_a_td_error = (q_target - q_eval).detach()
        # d_a_loss = (ISweights * torch.pow(q_eval - q_target.detach(), 2)).mean()

        # actor_loss = c_a_loss - c_actions_entropy.mean() - d_actions_entropy.mean()
        actor_loss = c_a_loss

        self.optimizer_a.zero_grad()
        actor_loss.backward()
        self.optimizer_a.step()

        actor_loss_r = actor_loss.item()

        new_p = critic_td_error

        self.memory.batch_update(choose_idx, new_p)

        return critic_loss_r, actor_loss_r


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
