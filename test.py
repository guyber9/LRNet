from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
import LRNet as my


def test():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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

    test_kwargs = {'batch_size': args.test_batch_size}

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    model = my.LRNet().to(device)
    model.load_state_dict(torch.load('./mnist_cnn.pt'))
    model.eval()

    print('Reading Database..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                           transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model.conv1.test_mode_switch()
    model.conv2.test_mode_switch()

    my.test(model, device, test_loader, True)


if __name__ == '__main__':
    test()