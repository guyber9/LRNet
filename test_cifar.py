from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
import LRNet as my
import train_cifar as my_cifar


def test():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
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

    model = my_cifar.FPNet_CIFAR10().to(device)
    model.load_state_dict(torch.load('./cifar10_full_prec.pt'))
    model.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**transform_test)
    test_loader = torch.utils.data.DataLoader(dataset2, **transform_test)

    print ("###################################")
    print ("Original Trained Model (no ternary)")
    print ("###################################")
    print ("test Data Set")
    my.test(model, device, test_loader, True)
    print ("train Data Set")
    my.test(model, device, train_loader, True)

    # model.conv1.test_mode_switch()
    # model.conv2.test_mode_switch()

    # print ("###################################")
    # print ("Ternary Model")
    # print ("###################################")
    # print ("test Data Set")
    # my.test(model, device, test_loader, True)
    # print ("train Data Set")
    # my.test(model, device, train_loader, True)


if __name__ == '__main__':
    test()