import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.Feature_embedding import Feature_Embedding


np.random.seed(1)

neuron_nums_1 = 100
neuron_nums_2 = 512
neuron_nums_3 = 256
neuron_nums_4 = 128


class Net(nn.Module):
    def __init__(self, field_nums, feature_nums, latent_dims, action_numbers, campaign_id):
        super(Net, self).__init__()
        self.field_nums = field_nums
        self.feature_nums = feature_nums
        self.latent_dims = latent_dims
        self.campaign_id = campaign_id

        input_dims = 0
        for i in range(self.field_nums):
            for j in range(i + 1, self.field_nums):
                input_dims += self.latent_dims
        input_dims += self.field_nums * self.latent_dims
        self.input_dims = input_dims

        self.fc1 = nn.Linear(self.input_dims, neuron_nums_1)
        self.fc1.weight.data.normal_(0, 0.1)  # 全连接隐层 1 的参数初始化
        self.fc2 = nn.Linear(neuron_nums_1, neuron_nums_2)
        self.fc2.weight.data.normal_(0, 0.1)  # 全连接隐层 2 的参数初始化
        self.fc3 = nn.Linear(neuron_nums_2, neuron_nums_3)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(neuron_nums_3, neuron_nums_4)
        self.fc4.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(neuron_nums_4, action_numbers)
        self.out.weight.data.normal_(0, 0.1)

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

        actions_value = torch.softmax(self.out(x_2), dim=1)

        return actions_value

class PolicyGradient:
    def __init__(
            self,
            feature_nums,
            field_nums,
            latent_dims,
            campaign_id,
            action_nums=2,
            learning_rate=1e-2,
            reward_decay=1,
            device='cuda:0',
    ):
        self.action_nums = action_nums
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims
        self.lr = learning_rate
        self.gamma = reward_decay
        self.device = device
        self.campaign_id = campaign_id

        self.ep_states, self.ep_as, self.ep_rs = [], [], [] # 状态，动作，奖励，在一轮训练后存储

        self.embedding_layer = Feature_Embedding(self.feature_nums, self.field_nums, self.latent_dims, self.campaign_id).to(self.device)

        self.policy_net = Net(self.field_nums, self.feature_nums, self.latent_dims, self.action_nums, self.campaign_id).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def load_embedding(self, pretrain_params):
        for i, embedding in enumerate(self.embedding_layer.field_feature_embeddings):
            embedding.weight.data.copy_(
                torch.from_numpy(
                    np.array(pretrain_params['field_feature_embeddings.' + str(i) + '.weight'].cpu())
                )
            )

    # def load_embedding(self, pretrain_params):
    #     self.embedding_layer.feature_embedding.weight.data.copy_(
    #         torch.from_numpy(
    #             np.array(pretrain_params['feature_embedding.weight'].cpu())
    #         )
    #     )

    def loss_func(self, all_act_prob, acts, vt):
        neg_log_prob = torch.sum(-torch.log(all_act_prob.gather(1, acts - 1)))
        loss = torch.mean(torch.mul(neg_log_prob, vt))
        return loss

    def choose_from_dribution(self, prob_weights):
        actions = []
        for prob_weight in prob_weights:
            random_a = np.random.choice(a=3, size=1, p=prob_weight.ravel())
            actions.append(random_a + 1)

        return np.array(actions)

    # 依据概率来选择动作，本身具有随机性
    def choose_action(self, state):
        state = self.embedding_layer(state)

        with torch.no_grad():
            prob_weights = self.policy_net.forward(state).detach().cpu().numpy()
            actions = self.choose_from_dribution(prob_weights)

        return actions

    def choose_best_action(self, state):
        state = self.embedding_layer(state)

        with torch.no_grad():
            prob_weights = self.policy_net.forward(state).detach().cpu().numpy()
            actions = []
            for prob_weight in prob_weights:
                actions.append(np.argmax(prob_weight) + 1)

        return np.array(actions).reshape(-1, 1)


    # 储存一回合的s,a,r；因为每回合训练
    def store_transition(self, s, a, r, i):
        if i % 81 == 0:
            self.ep_states = s
            self.ep_as = a
            self.ep_rs = r
        else:
            self.ep_states = np.vstack((self.ep_states, s))
            self.ep_as = np.vstack((self.ep_as, a))
            self.ep_rs = np.vstack((self.ep_rs, r))

    # 对每一轮的奖励值进行累计折扣及归一化处理
    def discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float64)
        running_add = 0
        # reversed 函数返回一个反转的迭代器。
        # 计算折扣后的 reward
        # 公式： E = r1 + r2 * gamma + r3 * gamma * gamma + r4 * gamma * gamma * gamma ...
        for i in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[i]
            discounted_ep_rs[i] = running_add

        # 归一化处理
        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 方差
        return discounted_ep_rs

    def learn(self):

        # 对每一回合的奖励，进行折扣计算以及归一化
        discounted_ep_rs_norm = self.discount_and_norm_rewards()

        states = self.embedding_layer(torch.LongTensor(self.ep_states).to(self.device))
        acts = torch.LongTensor(self.ep_as).to(self.device)
        vt = torch.FloatTensor(discounted_ep_rs_norm).to(self.device)

        all_act_probs = self.policy_net.forward(states)

        loss = self.loss_func(all_act_probs, acts, vt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.cuda.empty_cache()

        # 训练完后清除训练数据，开始下一轮
        self.ep_states, self.ep_as, self.ep_rs = [], [], []

        # return discounted_ep_rs_norm
