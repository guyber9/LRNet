import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np
import numpy as np
import myLRnet as my_nn

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    weight_decay = 1e-4
    probability_decay = 1e-11
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.cross_entropy(output, target)
        # loss = F.cross_entropy(output, target) + weight_decay * (torch.norm(model.fc1.weight, 2) + torch.norm(model.fc2.weight, 2)) \
        #             + probability_decay * (torch.norm(model.conv1.weight_theta, 2) + torch.norm(model.conv2.weight_theta, 2))

        if args.cifar10:
            loss = F.cross_entropy(output, target) + probability_decay * (torch.norm(model.conv1.alpha, 2) + torch.norm(model.conv1.betta, 2)
                                                 + torch.norm(model.conv2.alpha, 2) + torch.norm(model.conv2.betta, 2)
                                                 + torch.norm(model.conv3.alpha, 2) + torch.norm(model.conv3.betta, 2)
                                                 + torch.norm(model.conv4.alpha, 2) + torch.norm(model.conv4.betta, 2)
                                                 + torch.norm(model.conv5.alpha, 2) + torch.norm(model.conv5.betta, 2)
                                                 + torch.norm(model.conv6.alpha, 2) + torch.norm(model.conv6.betta, 2)) \
                                                 + weight_decay * (torch.norm(model.fc1.weight, 2) + (torch.norm(model.fc2.weight, 2)))
        else:
            loss = F.cross_entropy(output, target) + probability_decay * (torch.norm(model.conv1.alpha, 2)
                                                               + torch.norm(model.conv1.betta, 2)
                                                               + torch.norm(model.conv2.alpha, 2)
                                                               + torch.norm(model.conv2.betta, 2)) + weight_decay * (torch.norm(model.fc1.weight, 2) + (torch.norm(model.fc2.weight, 2)))

        if args.debug_mode:
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, test_mode):
    if test_mode:
        print ("evaluating with Test Model Parameters")
    else:
        print ("evaluating with Train Model Parameters")
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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


def print_full_tensor(input, input_name):
    for i, val1 in enumerate(input):
        for j, val2 in enumerate(val1):
            for m, val3 in enumerate(val2):
                print (str(input_name) + "(" + str(i) + ", " + str(j) + ", " + str(m) + ": " + str(val3))


def initialize_mnist (model, use_cuda, device, softmax_prob):
        test_model = my_nn.FPNet().to(device)
        if use_cuda:
            test_model.load_state_dict(torch.load('tmp_models/mnist_full_prec.pt'))
        else:
            test_model.load_state_dict(torch.load('tmp_models/mnist_full_prec_no_cuda.pt'))
        test_model.eval()
        # state_dict = torch.load('tmp_models/mnist_full_prec.pt')

        if softmax_prob:
            theta1 = find_weights(test_model.conv1.weight, False)
            theta2 = find_weights(test_model.conv2.weight, False)
            model.conv1.initialize_weights(theta1)
            model.conv2.initialize_weights(theta2)
        else:
            alpha1, betta1 = find_sigm_weights(test_model.conv1.weight, False)
            alpha2, betta2 = find_sigm_weights(test_model.conv2.weight, False)
            model.conv1.initialize_weights(alpha1, betta1)
            model.conv2.initialize_weights(alpha2, betta2)

        # model.conv1.bias = test_model.conv1.bias
        # model.conv2.bias = test_model.conv2.bias
        # model.fc1.weight = test_model.fc1.weight
        # model.fc1.bias = test_model.fc1.bias
        # model.fc2.weight = test_model.fc2.weight
        # model.fc2.bias = test_model.fc2.bias
        #
        # model.bn1.bias = test_model.bn1.bias
        # model.bn1.weight = test_model.bn1.weight
        # model.bn1.running_mean = test_model.bn1.running_mean
        # model.bn1.running_var = test_model.bn1.running_var
        #
        # model.bn2.bias = test_model.bn2.bias
        # model.bn2.weight = test_model.bn2.weight
        # model.bn2.running_mean = test_model.bn2.running_mean
        # model.bn2.running_var = test_model.bn2.running_var
        return model


def store_model(model, is_cifar10, full_prec, use_cuda):
      if is_cifar10:
        if full_prec:
            if use_cuda:
                torch.save(model.state_dict(), "tmp_models/cifar10_full_prec.pt")
            else:
                torch.save(model.state_dict(), "tmp_models/cifar10_full_prec_no_cuda.pt")
        else:
            torch.save(model.state_dict(), "tmp_models/cifar10_cnn.pt")
      else:
          if full_prec:
              if use_cuda:
                  torch.save(model.state_dict(), "tmp_models/mnist_full_prec.pt")
              else:
                  torch.save(model.state_dict(), "tmp_models/mnist_full_prec_no_cuda.pt")
          else:
              torch.save(model.state_dict(), "tmp_models/mnist_cnn.pt")