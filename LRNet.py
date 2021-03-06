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
import utils as utils
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

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
        # output = F.log_softmax(x, dim=1)
        # output = F.softmax(x)
        output = x
        return output


class LRNet(nn.Module):

    def __init__(self):
        super(LRNet, self).__init__()
        # self.conv1 = myConv2d(1, 32, 5, 1)
        # self.conv2 = myConv2d(32, 64, 5, 1)
        # self.conv1 = mySigmConv2d(1, 32, 5, 1)
        # self.conv2 = mySigmConv2d(32, 64, 5, 1)
        self.conv1 = MyNewConv2d(1, 32, 5, 1)
        self.conv2 = MyNewConv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv1.cuda(0)
        # self.conv1.cuda(1)
        # self.conv1 = nn.DataParallel(self.conv1, device_ids=[0, 1, 2])

    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
        x = self.bn1(x)
        # utils.print_full_tensor(x, "x_bn1")
        x = F.max_pool2d(x, 2) # 32 x 12 x 12
        x = F.relu(x)
        # x = self.dropout1(x) <= was here
        x = self.conv2(x) # 64 x 8 x 8
        x = self.bn2(x)
        # utils.print_full_tensor(x, "x_bn2")
        x = F.max_pool2d(x, 2) # 64 x 4 x 4
        x = F.relu(x)
        x = torch.flatten(x, 1) # 1024
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        # print("output" + str(output))
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

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # self.device = 'cpu'

        transposed = True
        if transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
            alpha = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, 1)
            betta = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, 1)
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size
            alpha = torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, 1)
            betta = torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, 1)

        # test_weight = torch.Tensor(D_0, D_1, D_2, D_3)
        # self.test_weight = nn.Parameter(test_weight)

        # self.alpha = nn.Parameter(alpha)
        # self.betta = nn.Parameter(betta)

        self.alpha = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=torch.float32, device=self.device))
        self.betta = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=torch.float32, device=self.device))
        self.test_weight = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3], dtype=torch.float32, device=self.device))

        # self.alpha = self.alpha.to(devic  e='cuda')
        # self.betta = self.betta.to(device='cuda')
        # self.test_weight = self.test_weight.to(device='cuda')

        # bias = torch.Tensor(out_channels)
        # self.bias = Parameter(bias)
        self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=torch.float, device=self.device))

        # self.discrete_val = torch.tensor([[-1.0, 0.0, 1.0]])
        # self.discrete_val.requires_grad = False
        # self.discrete_square_val = self.discrete_val * self.discrete_val
        # self.discrete_square_val.requires_grad = False
        #
        # discrete_mat = self.discrete_val.unsqueeze(1).repeat(D_0, D_1, D_2, D_3, 1)
        # discrete_square_mat = self.discrete_square_val.unsqueeze(1).repeat(D_0, D_1, D_2, D_3, 1)
        #
        # self.discrete_mat = nn.Parameter(torch.full(discrete_mat), dtype=torch.float, device=device)
        # self.discrete_square_mat = nn.Parameter(discrete_square_mat, dtype=torch.float, device=device)

        prob = torch.nn.Parameter(torch.tensor([-1.0, 0.0, 1.0], requires_grad=False, dtype=torch.float, device=self.device))
        prob_square = prob * prob
        self.discrete_mat = prob.repeat(D_0, D_1, D_2, D_3, 1)
        self.discrete_square_mat = prob_square.repeat(D_0, D_1, D_2, D_3, 1)

        # self.discrete_mat.requires_grad = False
        # self.discrete_square_mat.requires_grad = False

        print ("self.alpha: " + str(self.alpha.is_cuda))
        print ("self.betta: " + str(self.betta.is_cuda))
        print ("self.test_weight: " + str(self.test_weight.is_cuda))
        print ("self.bias: " + str(self.bias.is_cuda))
        print ("self.discrete_mat: " + str(self.discrete_mat.is_cuda))
        print ("self.discrete_square_mat: " + str(self.discrete_square_mat.is_cuda))

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
                        values_arr = np.random.default_rng().multinomial(500, np_theta)
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
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, device=self.device))
        self.betta = nn.Parameter(torch.tensor(betta, dtype=torch.float32, device=self.device))

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            #print("test_forward")
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            prob_alpha = self.sigmoid(self.alpha)
            prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)

            # print("################################################################")
            # print("START")
            # print("################################################################")
            # print(self.out_channels)
            # utils.print_full_tensor(prob_alpha, "prob_alpha")
            # utils.print_full_tensor(prob_betta, "prob_betta")

            prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
            # utils.print_full_tensor(prob_mat, "prob_mat")

            # if torch.cuda.is_available():
            #     prob_mat = prob_mat.to(device='cuda')

            # E[X] calc
            self.discrete_mat = self.discrete_mat.to(prob_mat.get_device())
            # print ("prob_mat: " + str(prob_mat.get_device()))
            # print ("self.discrete_mat: " + str(self.discrete_mat.get_device()))
            mean_tmp = prob_mat * self.discrete_mat
            mean = torch.sum(mean_tmp, dim=4)

            # print("mean: " + str(mean))

            # E[x^2]
            self.discrete_square_mat = self.discrete_square_mat.to(prob_mat.get_device())
            mean_square_tmp = prob_mat * self.discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)

            # E[x] ^ 2
            mean_pow2 = mean * mean

            # Var (E[x^2] - E[x]^2)
            # print("mean_tmp: " + str(mean_tmp))
            # print("mean_square_tmp: " + str(mean_square_tmp))
            # print("mean_pow2: " + str(mean_pow2))

            sigma_square = mean_square - mean_pow2

            # utils.print_full_tensor(mean_square, "mean_square")
            # utils.print_full_tensor(mean_pow2, "mean_pow2")
            # utils.print_full_tensor(sigma_square, "sigma_square")

            z0 = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)

            input_pow2 = (input * input)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                # devtype = dict(device=torch.device("cuda:0"), dtype=torch.float32)
                # input_pow2 = input_pow2.to(**devtype)
                # sigma_square = sigma_square.to(**devtype)
            z1 = F.conv2d(input_pow2, sigma_square, None, self.stride, self.padding, self.dilation, self.groups)

            epsilon = torch.rand(z1.size())
            if torch.cuda.is_available():
                epsilon = epsilon.to(device='cuda')
                epsilon = epsilon.to(z1.get_device())

            m = z0
            v = torch.sqrt(z1)
            # utils.print_full_tensor(m, "m")
            # utils.print_full_tensor(z1, "z1")
            # utils.print_full_tensor(v, "v")
            return m + epsilon * v

def train(args, model, device, train_loader, optimizer, epoch, f=None):
    model.train()
    weight_decay = 10**((-1)*args.wd) # 1e-4
    probability_decay = 10**((-1)*args.pd) # 1e-11
    print("weight_decay: " + str(weight_decay))
    print("probability_decay: " + str(probability_decay))
    # torch.backends.cudnn.benchmark = True
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # if args.parallel_gpu == 0:
        #     parallel_net = model
        # elif args.parallel_gpu == 1:
        #     parallel_net = nn.DataParallel(model, device_ids=[0])
        # elif args.parallel_gpu == 2:
        #     parallel_net = nn.DataParallel(model, device_ids=[0, 1])
        # elif args.parallel_gpu == 3:
        #     parallel_net = nn.DataParallel(model, device_ids=[0, 1, 2])
        # elif args.parallel_gpu == 4:
        #     parallel_net = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        # elif args.parallel_gpu == 5:
        #     dist.init_process_group(backend='nccl', rank=2, world_size=3, init_method = None, store = None)
        #     parallel_net = DDP(model, device_ids=[0, 1, 2], output_device=[0])
        # parallel_net = model

        optimizer.zero_grad()
        # output = parallel_net(data)
        output = model(data)

        if args.cifar10:
            if args.full_prec:
                loss = F.cross_entropy(output, target)
                ce_loss = loss
            else:
                ce_loss = F.cross_entropy(output, target)
                loss = ce_loss + probability_decay * (torch.norm(model.conv1.alpha, 2) + torch.norm(model.conv1.betta, 2)
                                                 + torch.norm(model.conv2.alpha, 2) + torch.norm(model.conv2.betta, 2)
                                                 + torch.norm(model.conv3.alpha, 2) + torch.norm(model.conv3.betta, 2)
                                                 + torch.norm(model.conv4.alpha, 2) + torch.norm(model.conv4.betta, 2)
                                                 + torch.norm(model.conv5.alpha, 2) + torch.norm(model.conv5.betta, 2)
                                                 + torch.norm(model.conv6.alpha, 2) + torch.norm(model.conv6.betta, 2)) \
                       + weight_decay * (torch.norm(model.fc1.weight, 2) + (torch.norm(model.fc2.weight, 2)))
                        # + weight_decay * (torch.norm(model.fc1.bias, 2) + (torch.norm(model.fc2.bias, 2)))
        # + weight_decay * (torch.norm(model.conv1.bias, 2) + torch.norm(model.conv2.bias, 2) \
        #                   + torch.norm(model.conv3.bias, 2) + torch.norm(model.conv4.bias, 2) \
        #                   + torch.norm(model.conv5.bias, 2) + torch.norm(model.conv6.bias, 2)) \
        else:
            if args.full_prec:
                loss = F.cross_entropy(output, target)
            else:
                loss = F.cross_entropy(output, target) + probability_decay * (torch.norm(model.conv1.alpha, 2)
                                                               + torch.norm(model.conv1.betta, 2)
                                                               + torch.norm(model.conv2.alpha, 2)
                                                               + torch.norm(model.conv2.betta, 2)) + weight_decay * (torch.norm(model.fc1.weight, 2) + (torch.norm(model.fc2.weight, 2)))
        # optimizer.zero_grad()

        if args.debug_mode:
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tce_loss: {:.6f}\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), ce_loss.item(), loss.item()))
        if f is not None:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tce_loss: {:.6f}\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item(), loss.item()), file = f)

            if args.dry_run:
                break

def test(model, device, test_loader, test_mode, f=None):
    if test_mode:
        tstring = 'Test'
    else:
        tstring = 'Train'
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n' + str(tstring) +' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if f is not None:
        print('\n' + str(tstring) + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), file = f)
    return (100. * correct / len(test_loader.dataset))


def find_weights(w, my_prints=False):
    if my_prints:
        print("w: " + str(w))
        print(w.size())
        print(type(w))
    # note: e^alpha + e^betta + e^gamma = 1
    p_max = 0.95
    p_min = 0.05
    w_norm = w / torch.std(w)
    e_alpha = p_max - ((p_max - p_min) * torch.abs(w_norm))
    if my_prints:
        print("e_alpha: " + str(e_alpha))
    e_alpha = torch.clamp(e_alpha, p_min, p_max)
    if my_prints:
        print("e_alpha.clip: " + str(e_alpha))
        print("e_alpha.size: " + str(e_alpha.size()))

    # betta = 0.5 * (1 + (w_norm / (1 - alpha)))
    e_betta = 0.5 * (w_norm - e_alpha + 1)
    if my_prints:
        print("e_betta: " + str(e_betta))
    e_betta = torch.clamp(e_betta, p_min, p_max)
    if my_prints:
        print("e_betta.clip: " + str(e_betta))

    alpha_prob = torch.log(e_alpha)
    betta_prob = torch.log(e_betta)
    gamma_prob = torch.log(torch.clamp((1 - e_alpha - e_betta), p_min, p_max))
    if my_prints:
        print("alpha_prob: " + str(alpha_prob))
        print("betta_prob: " + str(betta_prob))
        print("gamma_prob: " + str(gamma_prob))
    alpha_prob = alpha_prob.detach().cpu().clone().numpy()
    betta_prob = betta_prob.detach().cpu().clone().numpy()
    gamma_prob = gamma_prob.detach().cpu().clone().numpy()
    alpha_prob = np.expand_dims(alpha_prob, axis=-1)
    betta_prob = np.expand_dims(betta_prob, axis=-1)
    gamma_prob = np.expand_dims(gamma_prob, axis=-1)
    theta = np.concatenate((alpha_prob, betta_prob, gamma_prob), axis=4)
    if my_prints:
        print("theta: " + str(theta))
        print("theta.shape: " + str(np.shape(theta)))
    return theta


def find_sigm_weights(w, my_prints=False):
    if my_prints:
        print("w: " + str(w))
        print(w.size())
        print(type(w))

    p_max = 0.95
    p_min = 0.05
    w_norm = w / torch.std(w)
    e_alpha = p_max - ((p_max - p_min) * torch.abs(w_norm))
    e_betta = 0.5 * (1 + (w_norm / (1 - e_alpha)))
    if my_prints:
        print("alpha: " + str(alpha))
    e_alpha = torch.clamp(e_alpha, p_min, p_max)
    if my_prints:
        print("alpha.clip: " + str(e_alpha))
        print("alpha.size: " + str(e_alpha.size()))

    if my_prints:
        print("e_betta: " + str(e_betta))
    e_betta = torch.clamp(e_betta, p_min, p_max)
    if my_prints:
        print("e_betta.clip: " + str(e_betta))

    alpha_prob = torch.log(e_alpha / (1 - e_alpha))
    betta_prob = torch.log(e_betta / (1 - e_betta))
    if my_prints:
        print("alpha_prob: " + str(alpha_prob))
        print("betta_prob: " + str(betta_prob))
    alpha_prob = alpha_prob.detach().cpu().clone().numpy()
    betta_prob = betta_prob.detach().cpu().clone().numpy()
    alpha_prob = np.expand_dims(alpha_prob, axis=-1)
    betta_prob = np.expand_dims(betta_prob, axis=-1)
    return alpha_prob, betta_prob


class MyNewConv2d(nn.Module):

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
        transposed: bool = True,
        test_forward: bool = False,
    ):
        super(MyNewConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward
        self.transposed = transposed
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # transposed = True
        if self.transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size

        self.tensoe_dtype = torch.float32

        self.alpha = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensoe_dtype, device=self.device))
        self.betta = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensoe_dtype, device=self.device))
        self.test_weight = torch.empty([D_0, D_1, D_2, D_3], dtype=torch.float32, device=self.device)
        self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensoe_dtype, device=self.device))

        # discrete_prob = np.array([-1.0, 0.0, 1.0])
        # prob_mat = np.tile(discrete_prob, [D_0, D_1, D_2, D_3, 1])
        # square_prob_mat = prob_mat * prob_mat
        # self.discrete_mat = torch.tensor(prob_mat, requires_grad=False, dtype=torch.float32, device=self.device)
        # self.discrete_square_mat = torch.tensor(square_prob_mat, requires_grad=False, dtype=torch.float32, device=self.device)
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
            prob_size = torch.cat(((1 - self.alpha - self.betta), self.alpha, self.betta), 4)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(prob_size)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def initialize_weights(self, alpha, betta) -> None:
        print ("Initialize Weights")
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.tensoe_dtype, device=self.device))
        self.betta = nn.Parameter(torch.tensor(betta, dtype=self.tensoe_dtype, device=self.device))

    def test_mode_switch(self) -> None:
        print ("test_mode_switch")
        self.test_forward = True;
        print("Initializing Test Weights: \n")
        sigmoid_func = torch.nn.Sigmoid()
        alpha_prob = sigmoid_func(self.alpha)
        betta_prob = sigmoid_func(self.betta)  * (1 - alpha_prob)
        prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)

        prob_mat = prob_mat.detach().cpu().clone().numpy()

        my_array = []
        for i, val_0 in enumerate(prob_mat):
            my_array_0 = []
            for j, val_1 in enumerate(val_0):
                my_array_1 = []
                for m, val_2 in enumerate(val_1):
                    my_array_2 = []
                    for n, val_3 in enumerate(val_2):
                        # theta = val_3
                        values_arr = np.random.default_rng().multinomial(10, val_3)
                        values = np.nanargmax(values_arr) - 1
                        # print ("val_3: " + str(val_3))
                        # print ("values: " + str(values))
                        my_array_2.append(values)
                    my_array_1.append(my_array_2)
                my_array_0.append(my_array_1)
            my_array.append(my_array_0)
        self.test_weight = torch.tensor(my_array, dtype=self.tensoe_dtype, device=self.device)

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            prob_alpha = self.sigmoid(self.alpha)
            prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
            prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
            # E[X] calc
            # TODO: self.discrete_mat = self.discrete_mat.to(prob_mat.get_device())

            discrete_prob = np.array([-1.0, 0.0, 1.0])
            discrete_prob = np.tile(discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
            # discrete_mat = torch.tensor(discrete_prob, requires_grad=False, dtype=torch.float32, device='cuda')
            discrete_mat = torch.as_tensor(discrete_prob, dtype=self.tensoe_dtype, device='cuda')

            mean_tmp = prob_mat * discrete_mat
            mean = torch.sum(mean_tmp, dim=4)
            m = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # E[x^2]
            # TODO: self.discrete_square_mat = self.discrete_square_mat.to(prob_mat.get_device())
            square_discrete_prob = np.array([1.0, 0.0, 1.0])
            square_discrete_prob = np.tile(square_discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
            # discrete_square_mat = torch.tensor(square_discrete_prob, requires_grad=False, dtype=torch.float32, device='cuda')
            discrete_square_mat = torch.as_tensor(square_discrete_prob, dtype=self.tensoe_dtype, device='cuda')

            mean_square_tmp = prob_mat * discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)
            # E[x] ^ 2
            mean_pow2 = mean * mean
            sigma_square = mean_square - mean_pow2
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
            z1 = F.conv2d((input * input), sigma_square, None, self.stride, self.padding, self.dilation, self.groups)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = False
            v = torch.sqrt(z1)

            epsilon = torch.rand(z1.size(), requires_grad=False, dtype=self.tensoe_dtype, device='cuda')

            # epsilon = torch.rand(z1.size())
            # if torch.cuda.is_available():
            #     epsilon = epsilon.to(device='cuda')
            #     # epsilon = epsilon.to(z1.get_device())
            return m + epsilon * v
