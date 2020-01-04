import torch
import torch.nn as nn
import torch.utils.data

import numpy as np

# 传统的预测点击率模型
class LR(nn.Module):
    def __init__(self,
                 feature_nums,
                 output_dim = 1):
        super(LR, self).__init__()
        self.linear = nn.Embedding(feature_nums, output_dim)
        nn.init.xavier_normal_(self.linear.weight.data)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        out = self.bias + torch.sum(self.linear(x), dim=1)
        pctrs = torch.sigmoid(out)

        return pctrs

class FM(nn.Module):
    def __init__(self,
                 feature_nums,
                 latent_dims,
                 output_dim=1):
        super(FM, self).__init__()
        self.linear = nn.Embedding(feature_nums, output_dim)
        nn.init.xavier_normal_(self.linear.weight.data)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        self.feature_embedding = nn.Embedding(feature_nums, latent_dims)
        nn.init.xavier_normal_(self.feature_embedding.weight.data)

    def forward(self, x):
        """
        :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
        :return: pctrs
        """
        linear_x = x

        second_x = self.feature_embedding(x)

        square_of_sum = torch.sum(second_x, dim=1) ** 2
        sum_of_square = torch.sum(second_x ** 2, dim=1)

        ix = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True) # 若keepdim值为True,则在输出张量中,除了被操作的dim维度值降为1,其它维度与输入张量input相同。

        out = self.bias + torch.sum(self.linear(linear_x), dim=1) + ix * 0.5
        pctrs = torch.sigmoid(out)

        return pctrs

class FFM(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(FFM, self).__init__()

        self.field_nums = field_nums

        self.linear = nn.Embedding(feature_nums, output_dim)
        nn.init.xavier_normal_(self.linear.weight.data)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        '''
         FFM 每一个field都有一个关于所有特征的embedding矩阵，例如特征age=14，有一个age对应field的隐向量，
         但是相对于country的field有一个其它的隐向量，以此显示出不同field的区别 
       '''
        self.field_feature_embeddings = nn.ModuleList([
            nn.Embedding(feature_nums, latent_dims) for _ in range(field_nums)
        ]) # 相当于建立一个field_nums * feature_nums * latent_dims的三维矩阵
        for embedding in self.field_feature_embeddings:
            nn.init.xavier_normal_(embedding.weight.data)

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        x_embedding = [self.field_feature_embeddings[i](x) for i in range(self.field_nums)]
        second_x = list()
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                second_x.append(x_embedding[j][:, i] * x_embedding[i][:, j])
        # 因此预先输入了x，所以这里field的下标就对应了feature的下标，例如下标x_embedding[i][:, j]，假设j=3此时j就已经对应x=[13, 4, 5, 33]中的33
        # 总共有n(n-1)/2种组合方式，n=self.field_nums

        second_x = torch.stack(second_x, dim=1) # torch.stack(), torch.cat() https://blog.csdn.net/excellent_sun/article/details/95175823

        out = self.bias + torch.sum(self.linear(x), dim=1) + torch.sum(torch.sum(second_x, dim=1), dim=1, keepdim = True)
        pctrs = torch.sigmoid(out)

        return pctrs

# 基于深度学习的点击率预测模型
class WideAndDeep(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(WideAndDeep, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        nn.init.xavier_normal_(self.linear.weight.data)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        self.embedding = nn.Embedding(self.feature_nums, self.latent_dims)

        deep_input_dims = self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = 512
        for i in range(4):
            layers.append(nn.Linear(deep_input_dims, neuron_nums))
            # layers.append(nn.BatchNorm1d(neuron_nums))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_nums
            neuron_nums = int(neuron_nums / 2)

        layers.append(nn.Linear(deep_input_dims, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
        :return: pctrs
        """
        embedding_input = self.embedding(x)

        out = self.bias + torch.sum(self.linear(x), dim=1) + self.mlp(embedding_input.view(-1, self.field_nums * self.latent_dims))

        return torch.sigmoid(out)

class InnerPNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(InnerPNN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        nn.init.xavier_normal_(self.linear.weight.data)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)

        deep_input_dims = self.field_nums * self.latent_dims + self.field_nums * (self.field_nums - 1) // 2
        layers = list()

        neuron_nums = 512
        for i in range(4):
            layers.append(nn.Linear(deep_input_dims, neuron_nums))
            # layers.append(nn.BatchNorm1d(neuron_nums))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_nums
            neuron_nums = int(neuron_nums / 2)

        layers.append(nn.Linear(deep_input_dims, 1))
        self.mlp = nn.Sequential(*layers)

    def load_embedding(self, pretrain_params):
        self.feature_embedding.weight.data.copy_(
            torch.from_numpy(
                np.array(pretrain_params['feature_embedding.weight'].cpu())
            )
        )

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        embedding_x = self.feature_embedding(x).detach()

        inner_product_vectors = torch.FloatTensor().cuda()
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                inner_product = torch.sum(torch.mul(embedding_x[:, i], embedding_x[:, j]), dim=1).view(-1, 1)
                inner_product_vectors = torch.cat([inner_product_vectors, inner_product], dim=1)

        # 內积之和
        cross_term = inner_product_vectors

        cat_x = torch.cat([embedding_x.view(-1, self.field_nums * self.latent_dims), cross_term], dim=1)

        out = self.mlp(cat_x)

        return torch.sigmoid(out)

class OuterPNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(OuterPNN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        nn.init.xavier_normal_(self.linear.weight.data)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)

        deep_input_dims = self.latent_dims + self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = 512
        for i in range(4):
            layers.append(nn.Linear(deep_input_dims, neuron_nums))
            # layers.append(nn.BatchNorm1d(neuron_nums))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_nums
            neuron_nums = int(neuron_nums / 2)

        layers.append(nn.Linear(deep_input_dims, 1))
        self.mlp = nn.Sequential(*layers)

        # kernel指的对外积的转换
        self.kernel = nn.Parameter(torch.ones((self.latent_dims, self.latent_dims)))
        nn.init.xavier_normal_(self.kernel.data)

    def load_embedding(self, pretrain_params):
        self.feature_embedding.weight.data.copy_(
            torch.from_numpy(
                np.array(pretrain_params['feature_embedding.weight'].cpu())
            )
        )

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        embedding_x = self.feature_embedding(x).detach()

        sum_embedding_x = torch.sum(embedding_x, dim=1).unsqueeze(1)
        outer_product = torch.mul(torch.mul(sum_embedding_x, self.kernel), sum_embedding_x)

        cross_item = torch.sum(outer_product, dim=1) # 降维

        cat_x = torch.cat([embedding_x.view(-1, self.field_nums * self.latent_dims), cross_item], dim=1)
        out = self.mlp(cat_x)

        return torch.sigmoid(out)

class DeepFM(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(DeepFM, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        nn.init.xavier_normal_(self.linear.weight.data)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        # FM embedding
        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_normal_(self.feature_embedding.weight.data)

        # MLP
        deep_input_dims = self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = 512
        for i in range(4):
            layers.append(nn.Linear(deep_input_dims, neuron_nums))
            # layers.append(nn.BatchNorm1d(neuron_nums))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_nums
            neuron_nums = int(neuron_nums / 2)

        layers.append(nn.Linear(deep_input_dims, 1))
        self.mlp = nn.Sequential(*layers)

    def to_fm(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: FM's output
        """
        linear_x = x

        second_x = self.feature_embedding(x)

        square_of_sum = torch.sum(second_x, dim=1) ** 2
        sum_of_square = torch.sum(second_x ** 2, dim=1)

        ix = torch.sum(square_of_sum - sum_of_square, dim=1,
                       keepdim=True)  # 若keepdim值为True,则在输出张量中,除了被操作的dim维度值降为1,其它维度与输入张量input相同。

        out = self.bias + torch.sum(self.linear(linear_x), dim=1) + ix * 0.5

        return out

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        embedding_x = self.feature_embedding(x)

        out = self.to_fm(x) + self.mlp(embedding_x.view(-1, self.field_nums * self.latent_dims))

        return torch.sigmoid(out)

class FNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims):
        super(FNN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)

        deep_input_dims = self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = 512
        for i in range(4):
            layers.append(nn.Linear(deep_input_dims, neuron_nums))
            # layers.append(nn.BatchNorm1d(neuron_nums))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_nums
            neuron_nums = int(neuron_nums / 2)

        layers.append(nn.Linear(deep_input_dims, 1))
        self.mlp = nn.Sequential(*layers)

    def load_embedding(self, pretrain_params):
        self.feature_embedding.weight.data.copy_(
            torch.from_numpy(
                np.array(pretrain_params['feature_embedding.weight'].cpu())
            )
        )

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        embedding_x = self.feature_embedding(x).detach()
        out = self.mlp(embedding_x.view(-1, self.field_nums * self.latent_dims))

        return torch.sigmoid(out)