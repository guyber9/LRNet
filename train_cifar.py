from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import LRNet as my
from torch.nn import functional as F
import torch.nn as nn

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
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x) # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 128 x 16 x 16
        x = F.relu(x)
        x = self.dropout1(x)


        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x) # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2) # 256 x 8 x 8
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x) # 512 x 8 x 8
        x = self.bn6(x)
        x = F.max_pool2d(x, 2) # 512 x 4 x 4 (= 8192)
        x = F.relu(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1) # 8192
        # x = self.dropout2(x)
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
        x = self.fc2(x) # 1024 -> 10
        output = x
        return output


class LRNet(nn.Module):

    def __init__(self):
        super(LRNet, self).__init__()
        self.conv1 = my.myConv2d(1, 32, 5, 1)
        self.conv2 = my.myConv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
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
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
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
    parser.add_argument('--debug-mode', action='store_true', default=False,
                        help='For Debug Mode')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    print('Reading Database..')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)

    dataset2 = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.full_prec:
        print ("Training Net")
        model = FPNet_CIFAR10().to(device)
    else:
        print ("Training LRNet")
        model = LRNet().to(device)

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.load_pre_trained:
        test_model = my.FPNet_CIFAR10().to(device)
        if use_cuda:
            test_model.load_state_dict(torch.load('./cifar10_full_prec.pt'))
        else:
            test_model.load_state_dict(torch.load('./cifar10_full_prec_no_cuda.pt'))
        test_model.eval()
        # state_dict = torch.load('./cifar10_full_prec.pt')

        theta1 = my.find_weights(test_model.conv1.weight, False)
        theta2 = my.find_weights(test_model.conv2.weight, False)

        model.conv1.initialize_weights(theta1)
        model.conv2.initialize_weights(theta2)

        # model.conv1.bias.copy_(state_dict['conv1.bias'])
        # model.conv2.bias.copy_(state_dict['conv2.bias'])
        # model.fc1.weight.copy_(state_dict['fc1.weight'])
        # model.fc1.bias.copy_(state_dict['fc1.bias'])
        # model.fc2.weight.copy_(state_dict['fc2.weight'])
        # model.fc2.bias.copy_(state_dict['fc2.bias'])

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

    print ("###################################")
    print ("training..")
    print ("num of epochs: " + str(args.epochs))
    print ("###################################")
    my_step_size = 30
    scheduler = StepLR(optimizer, step_size=my_step_size, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        my.train(args, model, device, train_loader, optimizer, epoch)
        my.test(model, device, test_loader, False)
        if epoch == my_step_size:
            my.test(model, device, train_loader, False)
        # my.test(model, device, test_loader, True)
        scheduler.step()

    if args.full_prec:
        if use_cuda:
            torch.save(model.state_dict(), "cifar10_full_prec.pt")
        else:
            torch.save(model.state_dict(), "cifar10_full_prec_no_cuda.pt")
    else:
        torch.save(model.state_dict(), "cifar10_cnn.pt")

if __name__ == '__main__':
    main()
