import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from torch.nn.modules.conv import _single, _pair, _triple, _reverse_repeat_tuple
import numpy as np
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

class FPNet(nn.Module):

    def __init__(self):
        super(FPNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
        x = self.bn1(x)
        x = F.max_pool2d(x, 2) # 32 x 12 x 12
        x = F.relu(x)
        # x = self.dropout1(x)
        x = self.conv2(x) # 64 x 8 x 8
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 64 x 4 x 4
        x = F.relu(x)
        x = torch.flatten(x, 1) # 1024
        x = self.dropout2(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x
        return output


class LRNet(nn.Module):

    def __init__(self):
        super(LRNet, self).__init__()
        # self.conv1 = myConv2d(1, 32, 5, 1)
        # self.conv2 = myConv2d(32, 64, 5, 1)
        self.conv1 = mySigmConv2d(1, 32, 5, 1)
        self.conv2 = mySigmConv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
        x = self.bn1(x)
        x = F.max_pool2d(x, 2) # 32 x 12 x 12
        x = F.relu(x)
        # x = self.dropout1(x) <= was here
        x = self.conv2(x) # 64 x 8 x 8
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 64 x 4 x 4
        x = F.relu(x)
        x = torch.flatten(x, 1) # 1024
        x = self.dropout2(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x) # move to here tmp
        x = self.fc2(x)
        output = x
        return output


class myConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        clusters: int = 3,
        test_forward: bool = False,
    ):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward

        transposed = True
        if transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
            weight_theta = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, clusters)
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size
            weight_theta = torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, clusters)
        test_weight = torch.Tensor(D_0, D_1, D_2, D_3)
        self.test_weight = nn.Parameter(test_weight)
        self.weight_theta = nn.Parameter(test_weight)
        self.weight_theta = nn.Parameter(weight_theta)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(out_channels)
        self.bias = Parameter(bias)
        self.discrete_val = torch.tensor([[-1.0, 0.0, 1.0]])
        self.discrete_val.requires_grad = False
        # self.discrete_square_val = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
        self.discrete_square_val = self.discrete_val * self.discrete_val
        self.discrete_square_val.requires_grad = False

        if transposed:
            discrete_mat = self.discrete_val.unsqueeze(1).repeat(out_channels, in_channels, kernel_size, kernel_size, 1)
            discrete_square_mat = self.discrete_square_val.unsqueeze(1).repeat(out_channels, in_channels, kernel_size, kernel_size, 1)
        else:
            discrete_mat = self.discrete_val.unsqueeze(1).repeat(in_channels, out_channels, kernel_size,kernel_size, 1)
            discrete_square_mat = self.discrete_square_val.unsqueeze(1).repeat(out_channels, in_channels, kernel_size, kernel_size, 1)
        self.discrete_mat = nn.Parameter(discrete_mat)
        self.discrete_square_mat = nn.Parameter(discrete_square_mat)

        self.discrete_mat.requires_grad = False
        self.discrete_square_mat.requires_grad = False

        self.softmax = torch.nn.Softmax(dim=4)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # self.reset_train_parameters()
        init.kaiming_uniform_(self.weight_theta, a=math.sqrt(5))
        # init.uniform_(self.weight_theta, -1, 1)
        # init.constant_(self.weight_theta, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.discrete_mat)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def initialize_weights(self, theta) -> None:
        print ("Initialize Weights")
        self.weight_theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))

    def test_mode_switch(self) -> None:
        print("test_mode_switch")
        self.test_forward = True;
        print("Initializing Test Weights: \n")
        my_array = []
        softmax_func = torch.nn.Softmax()
        for i, val_0 in enumerate(self.weight_theta):
            my_array_0 = []
            for j, val_1 in enumerate(val_0):
                my_array_1 = []
                for m, val_2 in enumerate(val_1):
                    my_array_2 = []
                    for n, val_3 in enumerate(val_2):
                        theta = softmax_func(val_3)
                        # values = torch.multinomial(theta, 1) - 1
                        # values = torch.argmax(theta) - 1
                        if torch.cuda.is_available():
                            np_theta = theta.detach().cpu().clone().numpy().tolist()
                        else:
                            np_theta = theta.detach().numpy().tolist()
                        values_arr = np.random.default_rng().multinomial(1000, np_theta)
                        values = np.nanargmax(values_arr) - 1
                        my_array_2.append(values)
                    my_array_1.append(my_array_2)
                my_array_0.append(my_array_1)
            my_array.append(my_array_0)
        test_weight = torch.tensor(my_array, dtype=torch.float32)
        if torch.cuda.is_available():
            test_weight = test_weight.to(device='cuda')
        self.test_weight = nn.Parameter(test_weight)

    def reset_train_parameters(self) -> None:
        print("Initializing Train Weights: \n")
        p_max = 0.95
        p_min = 0.05
        wl = 0.7143
        p_w_0 = p_max - (p_max - p_min) * wl
        p_w_1 = 0.5 * (1 + (wl / (1 - p_w_0)))
        prob = np.array([(1 - p_w_1 - p_w_0), p_w_0, p_w_1])
        my_array = []
        for i, val_0 in enumerate(self.weight_theta):
            my_array_0 = [];
            for j, val_1 in enumerate(val_0):
                my_array_1 = [];
                for m, val_2 in enumerate(val_1):
                    my_array_2 = [];
                    for n, val_3 in enumerate(val_2):
                        my_array_2.append(prob)
                    my_array_1.append(my_array_2)
                my_array_0.append(my_array_1)
            my_array.append(my_array_0)
        weight_theta = torch.tensor(my_array, dtype=torch.float32)
        self.weight_theta = nn.Parameter(weight_theta)

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            # print("test_forward")
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # E[X] calc
            prob_mat = self.softmax(self.weight_theta)
            if torch.cuda.is_available():
                prob_mat = prob_mat.to(device='cuda')
            mean_tmp = prob_mat * self.discrete_mat
            mean = torch.sum(mean_tmp, dim=4)
            # E[x^2]
            mean_square_tmp = prob_mat * self.discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)
            # E[x] ^ 2
            mean_pow2 = mean * mean
            # Var (E[x^2] - E[x]^2)
            sigma_square = mean_square - mean_pow2
            z0 = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)
            z1 = F.conv2d((input * input), sigma_square, None, self.stride, self.padding, self.dilation, self.groups)
            epsilon = torch.rand(z1.size())
            if torch.cuda.is_available():
                epsilon = epsilon.to(device='cuda')
            m = z0
            v = torch.sqrt(z1)
            return m + epsilon * v

class mySigmConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        clusters: int = 3,
        test_forward: bool = False,
    ):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward

        transposed = True
        if transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
            alpha = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, 1)
            betta = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, 1)
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size
            alpha = torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, 1)
            betta = torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, 1)

        test_weight = torch.Tensor(D_0, D_1, D_2, D_3)
        self.test_weight = nn.Parameter(test_weight)

        self.alpha = nn.Parameter(alpha)
        self.betta = nn.Parameter(betta)

        bias = torch.Tensor(out_channels)
        self.bias = Parameter(bias)

        self.discrete_val = torch.tensor([[-1.0, 0.0, 1.0]])
        self.discrete_val.requires_grad = False
        self.discrete_square_val = self.discrete_val * self.discrete_val
        self.discrete_square_val.requires_grad = False

        if transposed:
            discrete_mat = self.discrete_val.unsqueeze(1).repeat(out_channels, in_channels, kernel_size, kernel_size, 1)
            discrete_square_mat = self.discrete_square_val.unsqueeze(1).repeat(out_channels, in_channels, kernel_size,
                                                                               kernel_size, 1)
        else:
            discrete_mat = self.discrete_val.unsqueeze(1).repeat(in_channels, out_channels, kernel_size, kernel_size, 1)
            discrete_square_mat = self.discrete_square_val.unsqueeze(1).repeat(out_channels, in_channels, kernel_size,
                                                                               kernel_size, 1)
        self.discrete_mat = nn.Parameter(discrete_mat)
        self.discrete_square_mat = nn.Parameter(discrete_square_mat)

        self.discrete_mat.requires_grad = False
        self.discrete_square_mat.requires_grad = False

        self.softmax = torch.nn.Softmax(dim=4)
        self.sigmoid = torch.nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # self.reset_train_parameters()
        init.constant_(self.alpha, -0.69314)
        init.constant_(self.betta, 0.0)
        # init.kaiming_uniform_(self.alpha, a=math.sqrt(5))
        # init.kaiming_uniform_(self.betta, a=math.sqrt(5))
        # init.uniform_(self.weight_theta, -1, 1)
        # init.constant_(self.weight_theta, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.discrete_mat)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def test_mode_switch(self) -> None:
        print ("test_mode_switch")
        self.test_forward = True;
        print("Initializing Test Weights: \n")
        sigmoid_func = torch.nn.Sigmoid()
        alpha_prob = sigmoid_func(self.alpha)
        betta_prob = sigmoid_func(self.betta)  * (1 - alpha_prob)
        prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
        my_array = []
        for i, val_0 in enumerate(prob_mat):
            my_array_0 = []
            for j, val_1 in enumerate(val_0):
                my_array_1 = []
                for m, val_2 in enumerate(val_1):
                    my_array_2 = []
                    for n, val_3 in enumerate(val_2):
                        theta = val_3
                        # print ("theta: " + str(theta))
                        # values = torch.multinomial(theta, 1) - 1
                        # my_array_2.append(values)
                        if torch.cuda.is_available():
                            np_theta = theta.detach().cpu().clone().numpy().tolist()
                        else:
                            np_theta = theta.detach().numpy().tolist()
                        values_arr = np.random.default_rng().multinomial(1, np_theta)
                        values = np.nanargmax(values_arr) - 1
                        my_array_2.append(values)
                    my_array_1.append(my_array_2)
                my_array_0.append(my_array_1)
            my_array.append(my_array_0)
        test_weight = torch.tensor(my_array, dtype=torch.float32)
        if torch.cuda.is_available():
            test_weight = test_weight.to(device='cuda')
        self.test_weight = nn.Parameter(test_weight)

    def initialize_weights(self, alpha, betta) -> None:
        print ("Initialize Weights")
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.betta = nn.Parameter(torch.tensor(betta, dtype=torch.float32))

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            prob_alpha = self.sigmoid(self.alpha)
            prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
            prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
            if torch.cuda.is_available():
                prob_mat = prob_mat.to(device='cuda')
            # E[X] calc
            mean_tmp = prob_mat * self.discrete_mat
            mean = torch.sum(mean_tmp, dim=4)
            # E[x^2]
            mean_square_tmp = prob_mat * self.discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)
            # E[x] ^ 2
            mean_pow2 = mean * mean
            sigma_square = mean_square - mean_pow2
            z0 = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)
            input_pow2 = (input * input)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                devtype = dict(device=torch.device("cuda"), dtype=torch.float32)
                input_pow2 = input_pow2.to(**devtype)
                sigma_square = sigma_square.to(**devtype)
            z1 = F.conv2d(input_pow2, sigma_square, None, self.stride, self.padding, self.dilation, self.groups)
            epsilon = torch.rand(z1.size())
            if torch.cuda.is_available():
                epsilon = epsilon.to(device='cuda')
            m = z0
            v = torch.sqrt(z1)
            return m + epsilon * v