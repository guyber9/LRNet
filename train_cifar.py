from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import LRNet as my
from torch.nn import functional as F
import torch.nn as nn
import utils as utils
import time

class FPNet_CIFAR10(nn.Module):

    def __init__(self):
        super(FPNet_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)
        # self.dropout3 = nn.Dropout(0.4)
        # self.dropout4 = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 3 
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x) # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 128 x 16 x 16
        x = F.relu(x)
        # x = self.dropout3(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x) # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2) # 256 x 8 x 8
        x = F.relu(x)
        # x = self.dropout4(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x) # 512 x 8 x 8
        x = self.bn6(x)
        x = F.max_pool2d(x, 2) # 512 x 4 x 4 (= 8192)
        x = F.relu(x)

        x = torch.flatten(x, 1) # 8192
        x = self.dropout1(x)
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x) # 1024 -> 10
        output = x
        return output


class LRNet_CIFAR10(nn.Module):

    def __init__(self):
        super(LRNet_CIFAR10, self).__init__()
        # self.conv1 = my.mySigmConv2d(3, 128, 3, 1, padding=1)
        # self.conv2 = my.mySigmConv2d(128, 128, 3, 1, padding=1)
        # self.conv3 = my.mySigmConv2d(128, 256, 3, 1, padding=1)
        # self.conv4 = my.mySigmConv2d(256, 256, 3, 1, padding=1)
        # self.conv5 = my.mySigmConv2d(256, 512, 3, 1, padding=1)
        # self.conv6 = my.mySigmConv2d(512, 512, 3, 1, padding=1)
        self.conv1 = my.MyNewConv2d(3, 128, 3, 1, padding=1)
        self.conv2 = my.MyNewConv2d(128, 128, 3, 1, padding=1)
        self.conv3 = my.MyNewConv2d(128, 256, 3, 1, padding=1)
        self.conv4 = my.MyNewConv2d(256, 256, 3, 1, padding=1)
        self.conv5 = my.MyNewConv2d(256, 512, 3, 1, padding=1)
        self.conv6 = my.MyNewConv2d(512, 512, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)
        # self.dropout3 = nn.Dropout(0.5)
        # self.dropout4 = nn.Dropout(0.5)
        # self.dropout5 = nn.Dropout(0.2)
        # self.dropout6 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        # utils.print_full_tensor(x, "x1")
        x = self.bn1(x)
        # utils.print_full_tensor(x, "x_bn1")
        x = F.relu(x)
        # x = self.dropout6(x)
        x = self.conv2(x)  # 128 x 32 x 32
        # utils.print_full_tensor(x, "x2")
        x = self.bn2(x)
        # utils.print_full_tensor(x, "x_bn2")
        x = F.max_pool2d(x, 2)  # 128 x 16 x 16
        x = F.relu(x)
        # x = self.dropout3(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.dropout5(x)
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)  # 256 x 8 x 8
        x = F.relu(x)
        # x = self.dropout4(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)  # 512 x 8 x 8
        x = self.bn6(x)
        x = F.max_pool2d(x, 2)  # 512 x 4 x 4 (= 8192)
        x = F.relu(x)

        x = torch.flatten(x, 1)  # 8192
        x = self.dropout1(x)
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        return output

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=100, metavar='M',
                        help='Step size for scheduler (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--full-prec', action='store_true', default=False,
                        help='For Training Full Precision Model')
    parser.add_argument('--load-pre-trained', action='store_true', default=False,
                        help='For Loading Params from Trained Full Precision Model')
    parser.add_argument('--debug-mode', action='store_true', default=False, help='For Debug Mode')
    parser.add_argument('--cifar10', action='store_true', default=True, help='cifar10 flag')
    parser.add_argument('--resume', action='store_true', default=False, help='resume model flag')
    parser.add_argument('--parallel-gpu', type=int, default=1, metavar='N',
                        help='parallel-gpu (default: 1)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='num_workers (default: 1)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    print('Reading Database..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset1 = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)

    dataset2 = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.full_prec:
        print ("Training FPNet_CIFAR10")
        model = FPNet_CIFAR10().to(device)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    else:
        print ("Training LRNet")
        model = LRNet_CIFAR10().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = optim.Adam([
        #     {'params': model.conv1.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv2.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv3.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv4.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv5.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv6.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.fc1.parameters(), 'weight_decay': 1e-4},
        #     {'params': model.fc2.parameters(), 'weight_decay': 1e-4}
        # ], lr=args.lr)

        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    if args.load_pre_trained:
        test_model = FPNet_CIFAR10().to(device)
        if use_cuda:
            test_model.load_state_dict(torch.load('tmp_models/cifar10_full_prec.pt'))
        else:
            test_model.load_state_dict(torch.load('tmp_models/cifar10_full_prec_no_cuda.pt'))
        test_model.eval()
        # state_dict = torch.load('tmp_models/cifar10_full_prec.pt')

        alpha1, betta1 = my.find_sigm_weights(test_model.conv1.weight, False)
        alpha2, betta2 = my.find_sigm_weights(test_model.conv2.weight, False)
        alpha3, betta3 = my.find_sigm_weights(test_model.conv3.weight, False)
        alpha4, betta4 = my.find_sigm_weights(test_model.conv4.weight, False)
        alpha5, betta5 = my.find_sigm_weights(test_model.conv5.weight, False)
        alpha6, betta6 = my.find_sigm_weights(test_model.conv6.weight, False)

        model.conv1.initialize_weights(alpha1, betta1)
        model.conv2.initialize_weights(alpha2, betta2)
        model.conv3.initialize_weights(alpha3, betta3)
        model.conv4.initialize_weights(alpha4, betta4)
        model.conv5.initialize_weights(alpha5, betta5)
        model.conv6.initialize_weights(alpha6, betta6)

        model.conv1.bias = test_model.conv1.bias
        model.conv2.bias = test_model.conv2.bias
        model.fc1.weight = test_model.fc1.weight
        model.fc1.bias = test_model.fc1.bias
        model.fc2.weight = test_model.fc2.weight
        model.fc2.bias = test_model.fc2.bias

        model.bn1.bias = test_model.bn1.bias
        model.bn1.weight = test_model.bn1.weight
        model.bn1.running_mean = test_model.bn1.running_mean
        model.bn1.running_var = test_model.bn1.running_var

        model.bn2.bias = test_model.bn2.bias
        model.bn2.weight = test_model.bn2.weight
        model.bn2.running_mean = test_model.bn2.running_mean
        model.bn2.running_var = test_model.bn2.running_var

    if args.resume:
        print("Resume Model: LRNet")
        model.load_state_dict(torch.load('tmp_models/cifar10_cnn.pt'))

    print ("###################################")
    print ("training..")
    print ("num of epochs: " + str(args.epochs))
    print ("###################################")
    if args.full_prec:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # with torch.cuda.amp.autocast():
        my.train(args, model, device, train_loader, optimizer, epoch)
        print('{} seconds'.format(time.time() - t0))
        my.test(model, device, test_loader, True)
        if ((epoch % 30) == 0) or (epoch == args.epochs):
            print("Accuracy on train data:")
            # torch.save(model.state_dict(), "tmp_models/cifar10_interim_model.pt")
            my.test(model, device, train_loader, False)
        scheduler.step()

    if args.full_prec:
        if use_cuda:
            torch.save(model.state_dict(), "tmp_models/cifar10_full_prec.pt")
        else:
            torch.save(model.state_dict(), "tmp_models/cifar10_full_prec_no_cuda.pt")
    else:
        torch.save(model.state_dict(), "tmp_models/cifar10_cnn.pt")

if __name__ == '__main__':
    main()
