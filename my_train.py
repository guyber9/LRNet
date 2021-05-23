from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import myLRnet as my
import utils as utils

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
    parser.add_argument('--cifar10', action='store_true', default=False, help='cifar10 flag')

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

    if args.cifar10:
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
    else:
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
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        print ("Training LRNet")
        model = my.LRNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.load_pre_trained:
        model = utils.initialize_mnist(model, use_cuda, device, False)

    print ("###################################")
    print ("training..")
    print ("num of epochs: " + str(args.epochs))
    print ("###################################")
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        utils.train(args, model, device, train_loader, optimizer, epoch)
        utils.test(model, device, test_loader, False)
        if ((epoch % 20) == 0) or (epoch == args.epochs):
            print("Accuracy on train data:")
            torch.save(model.state_dict(), "tmp_models/mnist_interim_model.pt")
            utils.test(model, device, train_loader, False)
        scheduler.step()

    utils.store_model(model, args.cifar10, args.full_prec, use_cuda)


if __name__ == '__main__':
    main()
