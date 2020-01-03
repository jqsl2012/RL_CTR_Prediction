import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.Feature_embedding import Feature_Embedding
np.seterr(all='raise')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

neuron_nums_1 = 100
neuron_nums_2 = 512
neuron_nums_3 = 256
neuron_nums_4 = 128


class Net(nn.Module):
    def __init__(self, field_nums, feature_nums, latent_dims, action_nums):
        super(Net, self).__init__()
        self.field_nums = field_nums
        self.feature_nums = feature_nums
        self.latent_dims = latent_dims
        self.embedding_layer = Feature_Embedding(self.feature_nums, self.field_nums, self.latent_dims)

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
        self.out = nn.Linear(neuron_nums_4, action_nums)
        self.out.weight.data.normal_(0, 0.1)

        self.dp = nn.Dropout(0.2)

    def forward(self, input):
        input = self.embedding_layer.forward(input)

        x = F.relu(self.fc1(input))
        x = self.dp(x)

        x_ = F.relu(self.fc2(x))
        x_ = self.dp(x_)

        x_1 = F.relu(self.fc3(x_))
        x_1 = self.dp(x_1)

        x_2 = F.relu(self.fc4(x_1))
        x_2 = self.dp(x_2)

        actions_value = self.out(x_2)

        return actions_value

# 定义Double DeepQNetwork
class DoubleDQN:
    def __init__(
            self,
            feature_nums,  # 状态的特征数量
            field_nums,
            latent_dims,
            action_nums=3,  # 动作的数量
            learning_rate=1e-3,  # 学习率
            reward_decay=1,  # 奖励折扣因子,偶发过程为1
            replace_target_iter=300,  # 每300步替换一次target_net的参数
            memory_size=500,  # 经验池的大小
            batch_size=32,  # 每次更新时从memory里面取多少数据出来，mini-batch
            device='cuda:0',
    ):
        self.action_nums = action_nums  # 动作的具体数值？[0,0.01,...,budget]
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.device = device

        # hasattr(object, name)
        # 判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
        # 需要注意的是name要用括号括起来
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 记录学习次数（用于判断是否替换target_net参数）
        self.learn_step_counter = 0

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = torch.zeros(size=[self.memory_size, self.field_nums + 2]).to(self.device)

        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.eval_net, self.target_net = Net(self.field_nums, self.feature_nums, self.latent_dims, self.action_nums).to(self.device), Net(
            self.field_nums, self.feature_nums, self.latent_dims, self.action_nums).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr, weight_decay=1e-5)

        # 损失函数为，均方损失函数
        self.loss_func = nn.MSELoss()

    # 经验池存储，s-state, a-action, r-reward
    def store_transition(self, transitions):
        transition_lens = len(transitions)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        index_end = (self.memory_counter + transition_lens) % self.memory_size

        if index_end > index_start:
            self.memory[index_start: index_end, :] = transitions  # 替换
        else:
            replace_len_1 = self.memory_size - index_start
            self.memory[index_start: self.memory_size, :] = transitions[0: replace_len_1]
            replace_len_2 = transition_lens - replace_len_1
            self.memory[0: replace_len_2, :] = transitions[replace_len_1: transition_lens]

        self.memory_counter += transition_lens

    # 选择动作
    def choose_action(self, states, exploration_rate):
        torch.cuda.empty_cache()

        actions = torch.LongTensor().to(self.device)

        action_values = self.eval_net.forward(states)
        for action_value in action_values:
            if np.random.uniform() > exploration_rate:
                # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
                action = torch.unsqueeze(torch.argsort(-action_value)[0] + 1, 0)
            else:
                action = torch.randint(low=1, high=self.action_nums+1, size=[1]).to(self.device)

            actions = torch.cat([actions, action])

        # 用矩阵来初始化

        return actions.view(-1, 1)

    # 选择最优动作
    def choose_best_action(self, states):
        action_values = self.eval_net.forward(states)

        actions = torch.argsort(-action_values, dim=1)[:, 0].view(-1, 1)

        return actions

    # 定义DQN的学习过程
    def learn(self):
        # 清除显存缓存
        torch.cuda.empty_cache()

        # 检查是否达到了替换target_net参数的步数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print(('\n目标网络参数已经更新\n'))
        self.learn_step_counter += 1

        # 训练过程
        # 从memory中随机抽取batch_size的数据
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)

        batch_memory = self.memory[sample_index, :].long()


        # 获取到q_next（target_net产生）以及q_eval（eval_net产生）
        # 如store_transition函数中存储所示，state存储在[0, feature_nums-1]的位置（即前feature_numbets）
        # state_存储在[feature_nums+1，memory_size]（即后feature_nums的位置）
        b_s = batch_memory[:, :self.field_nums]
        b_a = batch_memory[:, self.field_nums: self.field_nums + 1]
        b_r = batch_memory[:, self.field_nums + 1].view(-1, 1).float()
        b_s_ = batch_memory[:, :self.field_nums]

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net.forward(b_s).gather(1, b_a - 1)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next = self.target_net.forward(b_s_).detach()  # detach from graph, don't backpropagate，因为target网络不需要训练
        # 下一状态s的eval_net值
        q_eval_next = self.eval_net.forward(b_s_)
        max_b_a_next = torch.unsqueeze(torch.max(q_eval_next, 1)[1], 1)  # 选择最大Q的动作
        select_q_next = q_next.gather(1, max_b_a_next)

        q_target = b_r + self.gamma * select_q_next # shape (batch, 1)

        # 训练eval_net
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
