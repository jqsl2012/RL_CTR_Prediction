import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Categorical
import datetime
from torch.distributions import Normal, Categorical, MultivariateNormal

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Memory(object):
    def __init__(self, memory_size, transition_lens, device):
        self.device = device
        self.transition_lens = transition_lens # 存储的数据长度
        self.epsilon = 1e-3 # 防止出现zero priority
        self.alpha = 0.6 # 取值范围(0,1)，表示td error对priority的影响
        self.beta = 0.4 # important sample， 从初始值到1
        self.beta_increment_per_sampling = 1e-4
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
        p = td_error

        memory_start = self.memory_counter % self.memory_size
        memory_end = (self.memory_counter + len(transitions)) % self.memory_size

        if memory_end > memory_start:
            self.memory[memory_start: memory_end, :] = transitions
            self.prioritys_[memory_start: memory_end, :] = torch.max(self.prioritys_[memory_start: memory_end, :], p)
        else:
            replace_len_1 = self.memory_size - memory_start
            self.memory[memory_start: self.memory_size, :] = transitions[0: replace_len_1]
            self.prioritys_[memory_start: self.memory_size, :] = torch.max(self.prioritys_[memory_start: self.memory_size, :], p[0: replace_len_1])

            replace_len_2 = transition_lens - replace_len_1
            self.memory[:replace_len_2, :] = transitions[replace_len_1: transition_lens]
            self.prioritys_[:replace_len_2, :] = torch.max(self.prioritys_[:replace_len_2, :], p[replace_len_1: transition_lens])

        self.memory_counter += len(transitions)

    def stochastic_sample(self, batch_size):
        if self.memory_counter >= self.memory_size:
            priorities = self.get_priority(self.prioritys_)
            total_p = torch.sum(priorities, dim=0)
            min_prob = torch.min(priorities)
            # 采样概率分布
            P = torch.div(priorities, total_p).squeeze(1).cpu().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_size, batch_size, p=P, replace=False)).long().to(self.device)
        else:
            priorities = self.get_priority(self.prioritys_[:self.memory_counter, :])
            total_p = torch.sum(priorities, dim=0)
            min_prob = torch.min(priorities)
            P = torch.div(priorities, total_p).squeeze(1).cpu().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_counter, batch_size, p=P, replace=False)).long().to(self.device)

        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()

        batch = self.memory[sample_indexs]
        choose_priorities = priorities[sample_indexs]
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
        # p = self.get_priority(td_errors)
        self.prioritys_[choose_idx, :] = td_errors

def hidden_init(layer):
    # source: The other layers were initialized from uniform distributions
    # [− 1/sqrt(f) , 1/sqrt(f) ] where f is the fan-in of the layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Hybrid_Critic(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(Hybrid_Critic, self).__init__()
        self.input_dims = input_dims
        self.action_nums = action_nums

        deep_input_dims = self.input_dims + self.action_nums * 2

        neuron_nums = [512, 256]

        self.mlp_1 = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], 1)
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(3):
            if i % 2 == 0:
                self.mlp_1[i].weight.data.uniform_(*hidden_init(self.mlp_1[i]))
                self.mlp_2[i].weight.data.uniform_(*hidden_init(self.mlp_2[i]))

        self.mlp_1[4].weight.data.uniform_(-0.003, 0.003)
        self.mlp_2[4].weight.data.uniform_(-0.003, 0.003)

    def evaluate(self, input, c_actions, d_actions):
        obs = input
        c_q_out_1 = self.mlp_1(torch.cat([obs, c_actions, d_actions], dim=-1))
        c_q_out_2 = self.mlp_2(torch.cat([obs, c_actions, d_actions], dim=-1))

        return c_q_out_1, c_q_out_2

    def evaluate_q_1(self, input, c_actions, d_actions):
        obs = input

        c_q_out_1 = self.mlp_1(torch.cat([obs, c_actions, d_actions], dim=-1))

        return c_q_out_1

class C_Actor(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(C_Actor, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_nums

        neuron_nums = [512, 256]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dims + self.action_dims, neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], self.action_dims),
            nn.Tanh()
        )# 特征提取层

        self.reset_parameters()

        self.mean = torch.zeros(size=[1, 1]).cuda()
        self.std = torch.zeros(size=[1, 1]).cuda() * 0.1

    def reset_parameters(self):
        for i in range(3):
            if i % 2 == 0:
                self.mlp[i].weight.data.uniform_(*hidden_init(self.mlp[i]))

        self.mlp[4].weight.data.uniform_(-0.003, 0.003)

    def act(self, input, d_actions, temprature):
        obs = torch.cat([input, d_actions], dim=-1)
        c_action_means = self.mlp(obs)

        c_actions = torch.clamp(c_action_means + torch.normal(self.mean.expand_as(c_action_means), self.std.expand_as(c_action_means)), -1, 1)  # 用于返回训练
        ensemble_c_actions = torch.softmax(c_actions, dim=-1)

        return c_actions, ensemble_c_actions

    def evaluate(self, input, d_actions):
        obs = torch.cat([input, d_actions], dim=-1)
        c_actions_means = self.mlp(obs)

        return c_actions_means

class D_Actor(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(D_Actor, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_nums

        neuron_nums = [512, 256]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),

            nn.ReLU(),
            nn.Linear(neuron_nums[1], self.action_dims)
        )# 特征提取层

        self.reset_parameters()

        self.mean = torch.zeros(size=[1, 1]).cuda()
        self.std = torch.zeros(size=[1, 1]).cuda() * 0.2

    def reset_parameters(self):
        for i in range(3):
            if i % 2 == 0:
                self.mlp[i].weight.data.uniform_(*hidden_init(self.mlp[i]))

        self.mlp[4].weight.data.uniform_(-0.003, 0.003)

    def act(self, input, temprature):
        obs = input
        d_action_q_values = self.mlp(obs)
        d_action = gumbel_softmax_sample(logits=d_action_q_values + torch.normal(self.mean.expand_as(d_action_q_values), self.std.expand_as(d_action_q_values)), temperature=temprature, hard=False)
        ensemble_d_actions = torch.argmax(d_action, dim=-1) + 1

        return d_action, ensemble_d_actions.view(-1, 1)

    def evaluate(self, input):
        obs = input
        d_actions_q_values = self.mlp(obs)

        return d_actions_q_values

def boltzmann_softmax(actions, temprature):
    return (actions / temprature).exp() / torch.sum((actions / temprature).exp(), dim=-1).view(-1, 1)

def gumbel_softmax_sample(logits, temperature=1.0, hard=False, eps=1e-8, uniform_seed=1.0):
    U = Variable(torch.FloatTensor(*logits.shape).uniform_().cuda(), requires_grad=False)
    y = logits + -torch.log(-torch.log(U + eps) + eps)
    y = F.softmax(y / temperature, dim=-1)

    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y

    return y

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

class Hybrid_TD3_Model():
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
        self.action_nums = action_nums
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

        setup_seed(1)

        self.memory_counter = 0

        self.input_dims = self.field_nums * (self.field_nums - 1) // 2 + self.field_nums * self.latent_dims

        self.memory = Memory(self.memory_size, self.field_nums + self.action_nums * 2 + 2, self.device)

        self.C_Actor = C_Actor(self.input_dims, self.action_nums).to(self.device)
        self.D_Actor = D_Actor(self.input_dims, self.action_nums).to(self.device)
        self.Hybrid_Critic = Hybrid_Critic(self.input_dims, self.action_nums).to(self.device)

        self.C_Actor_ = copy.deepcopy(self.C_Actor)
        self.D_Actor_ = copy.deepcopy(self.D_Actor)
        self.Hybrid_Critic_ = copy.deepcopy(self.Hybrid_Critic)

        # 优化器
        self.optimizer_c_a = torch.optim.Adam(self.C_Actor.parameters(), lr=self.lr_C_A)
        self.optimizer_d_a = torch.optim.Adam(self.D_Actor.parameters(), lr=self.lr_C_A)
        self.optimizer_c = torch.optim.Adam(self.Hybrid_Critic.parameters(), lr=self.lr_C)

        self.loss_func = nn.MSELoss(reduction='mean')

        self.learn_iter = 0
        self.policy_freq = 2

        self.temprature = 3.0
        self.temprature_min = 1.0
        self.anneal_rate = 0.00005

        self.mean = torch.zeros(size=[1, 1]).to(self.device)
        self.std = torch.zeros(size=[1, 1]).to(self.device) * 0.4

    def store_transition(self, transitions): # 所有的值都应该弄成float
        if torch.max(self.memory.prioritys_) == 0.:
            td_errors = torch.ones(size=[len(transitions), 1]).to(self.device)
        else:
            td_errors = torch.max(self.memory.prioritys_).expand_as(torch.ones(size=[len(transitions), 1])).to(self.device)

        self.memory.add(td_errors, transitions)

    def choose_action(self, state):
        self.C_Actor.eval()
        self.D_Actor.eval()
        with torch.no_grad():
            # self.temprature = max(self.temprature, 0.01)
            d_q_values, ensemble_d_actions = self.D_Actor.act(state, self.temprature)
            c_actions, ensemble_c_actions = self.C_Actor.act(state, d_q_values, self.temprature)

        self.C_Actor.train()
        self.D_Actor.train()

        return c_actions, ensemble_c_actions, d_q_values, ensemble_d_actions

    def choose_best_action(self, state):
        self.C_Actor.eval()
        self.D_Actor.eval()
        with torch.no_grad():
            d_q_values = self.D_Actor.evaluate(state)
            c_action_means = self.C_Actor.evaluate(state, boltzmann_softmax(d_q_values, self.temprature))

        ensemble_c_actions = torch.softmax(c_action_means, dim=-1)
        ensemble_d_actions = torch.argsort(-d_q_values)[:, 0] + 1

        return ensemble_d_actions.view(-1, 1), ensemble_c_actions

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, embedding_layer):
        self.learn_iter += 1

        # sample
        choose_idx, batch_memory, ISweights = self.memory.stochastic_sample(self.batch_size)

        b_s = embedding_layer.forward(batch_memory[:, :self.field_nums].long())
        b_c_a = batch_memory[:, self.field_nums: self.field_nums + self.action_nums]
        b_d_a = batch_memory[:,
                self.field_nums + self.action_nums: self.field_nums + self.action_nums * 2]
        b_discrete_a = torch.unsqueeze(batch_memory[:, self.field_nums + self.action_nums * 2], 1)
        b_r = torch.unsqueeze(batch_memory[:, -1], 1)
        b_s_ = b_s  # embedding_layer.forward(batch_memory_states)

        with torch.no_grad():
            d_actions_q_values_next = self.D_Actor_.evaluate(b_s_)
            next_d_actions = gumbel_softmax_sample(logits=d_actions_q_values_next + torch.clamp(torch.normal(self.mean, self.std), -0.8, 0.8), temperature=self.temprature, hard=False)

            c_actions_means_next = self.C_Actor_.evaluate(b_s_, next_d_actions)
            next_c_actions = torch.clamp(c_actions_means_next + torch.clamp(torch.normal(self.mean, self.std), -0.8, 0.8), -1, 1)

            q1_target, q2_target = \
                self.Hybrid_Critic_.evaluate(b_s_, next_c_actions, next_d_actions)
            q_target = torch.min(q1_target, q2_target)
            q_target = b_r + self.gamma * q_target

        q1, q2 = self.Hybrid_Critic.evaluate(b_s, b_c_a, b_d_a)

        critic_td_error = (q_target * 2 - q1 - q1).detach() / 2

        critic_loss = (ISweights * (F.mse_loss(q1, q_target, reduction='none') + F.mse_loss(q2, q_target, reduction='none'))).mean()

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.Hybrid_Critic.parameters(), 0.5)
        self.optimizer_c.step()

        critic_loss_r = critic_loss.item()

        self.memory.batch_update(choose_idx, critic_td_error)

        if self.learn_iter % 10 == 0:
            self.temprature = max(self.temprature * np.exp(-self.anneal_rate * self.learn_iter), self.temprature_min)

        if self.learn_iter % self.policy_freq == 0:
            d_actions_q_values = self.D_Actor.evaluate(b_s)
            d_actions_q_values_ = gumbel_softmax_sample(d_actions_q_values, hard=False, temperature=self.temprature)

            c_actions_means = self.C_Actor.evaluate(b_s, d_actions_q_values_)

            # Hybrid_Actor
            # c a
            a_critic_value = self.Hybrid_Critic.evaluate_q_1(b_s, c_actions_means, d_actions_q_values_)
            c_a_loss = -a_critic_value.mean()

            self.optimizer_c_a.zero_grad()
            c_a_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.C_Actor.parameters(), 0.5)
            self.optimizer_c_a.step()

            self.optimizer_d_a.zero_grad()
            c_a_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.D_Actor.parameters(), 0.5)
            self.optimizer_d_a.step()

            self.soft_update(self.Hybrid_Critic, self.Hybrid_Critic_)
            self.soft_update(self.C_Actor, self.C_Actor_)
            self.soft_update(self.D_Actor, self.D_Actor_)

        return critic_loss_r

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
