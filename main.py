import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

import argparse
import os
from pprint import pprint

from utils import set_seed, transform_train, transform_test, make_and_restore_model, CIFAR10Poisoned
from train import train_model, eval_model


def main(args):
    
    if args.poison_type == 'C':
        data_set = datasets.CIFAR10(args.clean_data_path, train=True, download=True, transform=transform_train)
    else:
        data_set = CIFAR10Poisoned(args.poison_data_path, args.constraint, args.poison_type, transform=transform_train)
    
    set_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(data_set, [len(data_set) - args.val_num_examples, args.val_num_examples], generator=torch.Generator().manual_seed(args.seed))
    test_set = datasets.CIFAR10(args.clean_data_path, train=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    if not os.path.isfile(args.model_save_path):
        model = make_and_restore_model(args.arch)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
        writer = SummaryWriter(args.tensorboard_path)
        train_model(args, model, optimizer, schedule, train_loader, val_loader, test_loader, writer)
    
    model = make_and_restore_model(args.arch, resume_path=args.model_save_path)
    args.num_steps = 20
    args.step_size = args.eps * 2.5 / args.num_steps
    args.random_restarts = 5

    eval_model(args, model, val_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training classifiers for CIFAR10')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=0.1, choices=[0.1, 0.05, 0.01], type=float)
    parser.add_argument('--batch_size', default=128, choices=[128, 256], type=int)
    parser.add_argument('--weight_decay', default=5e-4, choices=[0, 1e-4, 5e-4, 1e-3], type=float)

    parser.add_argument('--train_loss', default='ST', choices=['ST', 'AT'], type=str)
    parser.add_argument('--eps', default=0.5, type=float)
    parser.add_argument('--constraint', default='L2', choices=['Linf', 'L2'], type=str)

    parser.add_argument('--arch', default='VGG16', type=str, choices=['VGG16', 'VGG19', 'ResNet18', 'ResNet50', 'DenseNet121'])
    
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--out_dir', default='results/', type=str)
    parser.add_argument('--clean_data_path', default='./datasets/CIFAR10', type=str)
    parser.add_argument('--poison_data_path', default='./datasets/CIFAR10Poison', type=str)
    parser.add_argument('--poison_type', default='C', choices=['C', 'P1', 'P2', 'P3', 'P4', 'P5'])

    parser.add_argument('--gpuid', default=0, type=int)

    args = parser.parse_args()
    
    args.poison_name = 'C' if args.poison_type == 'C' and args.train_loss == 'ST' else '{}({})'.format(args.poison_type, args.constraint)
    args.exp_name = '{}-{}on{}-lr{}-bs{}-wd{}-seed{}'.format(args.arch, args.train_loss, args.poison_name, args.lr, args.batch_size, args.weight_decay, args.seed)
    args.tensorboard_path = os.path.join(os.path.join(args.out_dir, args.exp_name), 'tensorboard')
    args.model_save_path = os.path.join(os.path.join(args.out_dir, args.exp_name), 'checkpoint.pth')
    args.epochs = 150
    args.lr_milestones = [100, 125]
    args.lr_step = 0.1
    args.log_gap = 1
    args.step_size = args.eps / 5
    args.num_steps = 7
    args.random_restarts = 1
    args.val_num_examples = 1000

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)

