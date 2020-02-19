import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.distributions import MultivariateNormal, Categorical

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1)

class Hybrid_Actor_Critic(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(Hybrid_Actor_Critic, self).__init__()
        self.input_dims = input_dims

        neuron_nums = [300, 300, 300]

        # Critic
        self.Critic = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
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

        # Continuous_Actor
        self.Continuous_Actor = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
            nn.Linear(neuron_nums[2], action_nums),
            nn.Softmax(dim=-1)
        )

        # Discrete_Actor
        self.Discrete_Actor = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
            nn.Linear(neuron_nums[2], action_nums - 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, input, type):
        if type == 'critic':
            state_value = self.Critic(input)
            return state_value
        elif type == 'c_a':
            c_action = self.Continuous_Actor(input) # no softmax
            return c_action
        elif type == 'd_a':
            d_action = self.Discrete_Actor(input) # no softmax
            return d_action
        else:
            raise ModuleNotFoundError


class Hybrid_RL_Model():
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
            memory_size=4096000, # 设置为DataLoader的batch_size * n
            batch_size=256,
            tau=0.005,  # for target network soft update
            k_epochs=3, # update policy for k epochs
            eps_clip=0.2, # clip parameter for ppo
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
        self.action_std = 0.5
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip

        self.memory_counter = 0

        self.input_dims = self.field_nums * (self.field_nums - 1) // 2 + self.field_nums * self.latent_dims

        self.memory_state = torch.zeros(size=[self.memory_size, self.field_nums]).to(self.device)
        self.memory_c_a = torch.zeros(size=[self.memory_size, self.action_nums]).to(self.device)
        self.memory_c_logprobs = torch.zeros(size=[self.memory_size, 1]).to(self.device)
        self.memory_d_a = torch.zeros(size=[self.memory_size, 1]).to(self.device)
        self.memory_d_logprobs = torch.zeros(size=[self.memory_size, 1]).to(self.device)
        self.memory_reward = torch.zeros(size=[self.memory_size, 1]).to(self.device)

        self.hybrid_actor_critic = Hybrid_Actor_Critic(self.input_dims, self.action_nums).to(self.device)

        self.hybrid_actor_critic_old = Hybrid_Actor_Critic(self.input_dims, self.action_nums).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.hybrid_actor_critic.parameters(), lr=self.lr_A, weight_decay=1e-5)

        self.loss_func = nn.MSELoss(reduction='mean')

    def store_memory(self, states, c_a, c_logprobs, d_a, d_logprobs, rewards):
        transition_lens = len(states)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        index_end = (self.memory_counter + transition_lens) % self.memory_size

        self.memory_state[index_start: index_end, :] = states
        self.memory_c_a[index_start: index_end, :] = c_a
        self.memory_c_logprobs[index_start: index_end, :] = c_logprobs
        self.memory_d_a[index_start: index_end, :] = d_a
        self.memory_d_logprobs[index_start: index_end, :] = d_logprobs
        self.memory_reward[index_start: index_end, :] = rewards

        self.memory_counter += transition_lens

    def evaluate_v(self, state):
        return self.hybrid_actor_critic.forward(state, 'critic')

    def choose_c_a(self, state):
        self.hybrid_actor_critic.eval()
        with torch.no_grad():
            action_mean = self.hybrid_actor_critic.forward(state, 'c_a')

        action_std = torch.diag(torch.full((self.action_nums,), self.action_std * self.action_std)).to(self.device)
        action_dist = MultivariateNormal(action_mean, action_std)

        actions = action_dist.sample() # 分布产生的结果
        actions_logprobs = action_dist.log_prob(actions).view(-1, 1)

        ensemble_c_actions = torch.softmax(actions, dim=-1) # 模型所需的动作

        self.hybrid_actor_critic.train()

        return actions, actions_logprobs, ensemble_c_actions

    def evaluate_c_a(self, state, action):
        self.hybrid_actor_critic.eval()
        with torch.no_grad():
            action_mean = self.hybrid_actor_critic.forward(state, 'c_a')

        action_std = torch.diag(torch.full((self.action_nums,), self.action_std * self.action_std)).to(self.device)
        action_dist = MultivariateNormal(action_mean, action_std)

        action_logprobs = action_dist.log_prob(action)

        return action_logprobs

    def choose_d_a(self, state):
        self.hybrid_actor_critic.eval()
        with torch.no_grad():
            action_values = self.hybrid_actor_critic.forward(state, 'd_a')

        action_dist = Categorical(action_values)
        actions = action_dist.sample() # 分布产生的结果

        actions_logprobs = action_dist.log_prob(actions).view(-1, 1)

        ensemble_d_actions = actions + 2 # 模型所需的动作

        self.hybrid_actor_critic.train()

        return actions.view(-1, 1), actions_logprobs, ensemble_d_actions

    def evaluate_d_a(self, state, action):
        self.hybrid_actor_critic.eval()
        with torch.no_grad():
            action_values = self.hybrid_actor_critic.forward(state, 'd_a')
        action_dist = Categorical(action_values)

        actions_logprobs = action_dist.log_prob(action)

        return actions_logprobs

    def choose_best_continuous_action(self, state):
        self.hybrid_actor_critic.eval()
        with torch.no_grad():
            ensemble_c_actions = self.hybrid_actor_critic.forward(state, 'c_a')

        return ensemble_c_actions

    def choose_best_discrete_action(self, state):
        self.hybrid_actor_critic.eval()
        with torch.no_grad():
            action_values = torch.softmax(self.hybrid_actor_critic.forward(state))

        ensemble_d_actions = torch.argsort(-action_values)[:, 0] + 2

        return ensemble_d_actions.view(-1, 1)

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

            # 对每一轮的奖励值进行累计折扣及归一化处理

    def learn(self):
        states = self.memory_state
        states_ = self.memory_state
        old_c_a = self.memory_c_a
        old_c_a_logprobs = self.memory_c_logprobs
        old_d_a = self.memory_d_a
        old_d_a_logprobs = self.memory_d_logprobs
        rewards = self.memory_reward

        for _ in range(self.k_epochs):
            value_of_states_ = self.evaluate_v(states_) # 下一状态的V值
            value_of_states = self.evaluate_v(states) # 当前状态的V值

            td_target = rewards + self.gamma * value_of_states_
            advantages = td_target - value_of_states

            # Normalizing the rewards
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # Update Continuous Actor
            c_a_logprobs = self.evaluate_c_a(states, old_c_a)
            ratios = torch.exp(c_a_logprobs - old_c_a_logprobs)
            c_a_surr1 = ratios * advantages
            c_a_surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            c_a_loss = -torch.min(c_a_surr1, c_a_surr2)

            # Update Discrete Actor
            d_a_logprobs = self.evaluate_d_a(states, old_d_a)
            ratios = torch.exp(d_a_logprobs - old_d_a_logprobs)
            d_a_surr1 = ratios * advantages
            d_a_surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            d_a_loss = -torch.min(d_a_surr1, d_a_surr2)

            # Update Value Layer(Critic)
            critic_loss = self.loss_func(value_of_states, td_target)

            loss = c_a_loss + d_a_loss + critic_loss

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.hybrid_actor_critic_old.load_state_dict(self.hybrid_actor_critic.state_dict())



