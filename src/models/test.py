import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical, Normal

x = torch.softmax(torch.Tensor([[0.1, 0.2, 0.7],[0.3, 0.4, 0.2]]), dim=-1)
print(torch.softmax(x, dim=-1))
print(torch.softmax(x, dim=1))

print(x)
y = torch.exp(torch.log(x))
print(y)

t = torch.diag(torch.full((3,), 1*1))
print(t)

o = 0.9
for i in range(300):
    if (i + 1) % 10 == 0:
        o -= 10/300
        print(o)

u = MultivariateNormal(x, t)
k = u.sample()
p = u.log_prob(k)
print(p)
print(k)
print(torch.softmax(k, dim=1))

t = torch.full((3,), 1*1).expand_as(x)
print(torch.diag_embed(t))

C = Categorical(torch.softmax(x, dim=1))
print(C.sample() + 2)

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
            nn.Linear(neuron_nums[2], action_nums)
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
            nn.Linear(neuron_nums[2], action_nums)
        )



Hybrid_Actor_Critic(10,2)

m = Normal(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0]))
k = m.sample()
o = m.sample()
l = m.log_prob(k)
l1 = m.log_prob(o)
print(k, l, o)

print(torch.mean((l - l1).exp()))

init_lr = 1e-2
end_lr = 1e-4
lr_lamda = (init_lr - end_lr) / 400

for i in range(500):
    print(max(init_lr - lr_lamda * (i + 1), end_lr))

beta = torch.min(torch.FloatTensor([1., 0.5 + 0.01]))
print(beta)

print(torch.min)

