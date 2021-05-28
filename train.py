from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import LRNet as my
import torch.nn as nn
import torch.distributed as dist

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
    parser.add_argument('--debug-mode', action='store_true', default=False,
                        help='For Debug Mode')
    parser.add_argument('--cifar10', action='store_true', default=False,
                        help='cifar10 flag')
    parser.add_argument('--parallel-gpu', type=int, default=1, metavar='N',
                        help='parallel-gpu (default: 1)')

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
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.full_prec:
        print ("Training Net")
        model = my.FPNet().to(device)
    else:
        print ("Training LRNet")
        model = my.LRNet().to(device)

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.load_pre_trained:
        test_model = my.FPNet().to(device)
        if use_cuda:
            test_model.load_state_dict(torch.load('tmp_models/mnist_full_prec.pt'))
        else:
            test_model.load_state_dict(torch.load('tmp_models/mnist_full_prec_no_cuda.pt'))
        test_model.eval()
        # state_dict = torch.load('tmp_models/mnist_full_prec.pt')

        # theta1 = my.find_weights(test_model.conv1.weight, False)
        # theta2 = my.find_weights(test_model.conv2.weight, False)

        # model.conv1.initialize_weights(theta1)
        # model.conv2.initialize_weights(theta2)

        alpha1, betta1 = my.find_sigm_weights(test_model.conv1.weight, False)
        alpha2, betta2 = my.find_sigm_weights(test_model.conv2.weight, False)

        model.conv1.initialize_weights(alpha1, betta1)
        model.conv2.initialize_weights(alpha2, betta2)

        # model.conv1.bias.copy_(state_dict['conv1.bias'])
        # model.conv2.bias.copy_(state_dict['conv2.bias'])
        # model.fc1.weight.copy_(state_dict['fc1.weight'])
        # model.fc1.bias.copy_(state_dict['fc1.bias'])
        # model.fc2.weight.copy_(state_dict['fc2.weight'])
        # model.fc2.bias.copy_(state_dict['fc2.bias'])

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

    if args.parallel_gpu:
        dist.init_process_group(backend='nccl')

    print ("###################################")
    print ("training..")
    print ("num of epochs: " + str(args.epochs))
    print ("###################################")
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        my.train(args, model, device, train_loader, optimizer, epoch)
        my.test(model, device, test_loader, True)
        if ((epoch % 20) == 0) or (epoch == args.epochs):
            print("Accuracy on train data:")
            torch.save(model.state_dict(), "tmp_models/mnist_interim_model.pt")
            my.test(model, device, train_loader, False)
        scheduler.step()

    if args.full_prec:
        if use_cuda:
            torch.save(model.state_dict(), "tmp_models/mnist_full_prec.pt")
        else:
            torch.save(model.state_dict(), "tmp_models/mnist_full_prec_no_cuda.pt")
    else:
        torch.save(model.state_dict(), "tmp_models/mnist_cnn.pt")

if __name__ == '__main__':
    main()
