import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import make_grid

import argparse
import os
from tqdm import tqdm
from pprint import pprint

from utils import set_seed, CIFAR10Poisoned, AverageMeter, accuracy_top1, transform_test, make_and_restore_model
from attacks.step import LinfStep, L2Step
from utils import show_image_row

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}


def batch_poison(model, x, target, args, adv_or_hyp):
    orig_x = x.clone().detach()
    step = STEPS[args.constraint](orig_x, args.eps, args.step_size)

    if adv_or_hyp == 'adv':
        target = (target + 1) % 10  # Using a fixed permutation of labels
    elif adv_or_hyp == 'hyp':
        target = target     # Maximize accuracy
    
    for _ in range(args.num_steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [x])[0]
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
            x = torch.clamp(x, 0, 1)
    
    return x.clone().detach().requires_grad_(False)


def poison_p1p2(args, loader, model, writer, adv_or_hyp):
    poisoned_input = []
    clean_target = []
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        inp_p = batch_poison(model, inp, target, args, adv_or_hyp)

        poisoned_input.append(inp_p.detach().cpu())
        clean_target.append(target.detach().cpu())

        with torch.no_grad():
            logits = model(inp_p)
            loss = nn.CrossEntropyLoss()(logits, target)
            acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))
        desc = ('[{} {}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(args.poison_type, args.eps, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)
    
    poisoned_input = torch.cat(poisoned_input, dim=0)
    clean_target = torch.cat(clean_target, dim=0)
    return poisoned_input, clean_target


def poison_p1(args, loader, model, writer):
    poisoned_data = poison_p1p2(args, loader, model, writer, adv_or_hyp='adv')
    return poisoned_data


def poison_p2(args, loader, model, writer):
    poisoned_data = poison_p1p2(args, loader, model, writer, adv_or_hyp='hyp')
    return poisoned_data


def poison_p3p4p5(args, loader, model, writer, poisons):
    poisoned_input = []
    clean_target = []
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Add the same perturbation to examples from the same class
        index = target.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand(-1, 3, 32, 32)
        delta = torch.gather(poisons, dim=0, index=index)

        inp_p = inp + delta
        inp_p = torch.clamp(inp_p, 0, 1)

        poisoned_input.append(inp_p.detach().cpu())
        clean_target.append(target.detach().cpu())

        with torch.no_grad():
            logits = model(inp_p)
            loss = nn.CrossEntropyLoss()(logits, target)
            acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))
        desc = ('[{} {}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(args.poison_type, args.eps, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)
    
    poisoned_input = torch.cat(poisoned_input, dim=0)
    clean_target = torch.cat(clean_target, dim=0)
    return poisoned_input, clean_target


def universal_target_attack(model, loader, target_class, writer, args):
    delta = torch.zeros(1, *args.data_shape).cuda(non_blocking=True)
    orig_delta = delta.clone().detach()
    step = STEPS[args.constraint](orig_delta, args.eps, args.step_size)

    tag = 'universal_perturbation/{}-{}'.format(target_class, loader.dataset.classes[target_class])
    vis = make_grid(delta, nrow=1, normalize=True)
    writer.add_image(tag, vis, global_step=0)

    data_loader = DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=True)
    data_iter = iter(data_loader)

    iterator = tqdm(range(args.num_steps * 5), total=args.num_steps * 5)
    for i in iterator:
        try:
            inp, target = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            inp, target = next(data_iter)

        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target_ori = target.clone()
        target.fill_(target_class)

        delta = delta.clone().detach().requires_grad_(True)
        inp_adv = inp + delta
        inp_adv = torch.clamp(inp_adv, 0, 1)
        logits = model(inp_adv)
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [delta])[0]

        with torch.no_grad():
            delta = step.step(delta, grad)
            delta = step.project(delta)
            acc = accuracy_top1(logits, target_ori)

        if writer is not None and i % 10 == 0:
            # Visualization
            tag = 'universal_perturbation/{}-{}'.format(target_class, loader.dataset.classes[target_class])
            vis = make_grid(delta, nrow=1, normalize=True)
            writer.add_image(tag, vis, global_step=i+1)

        desc = ('[ Target class {}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(target_class, loss.item(), acc))
        iterator.set_description(desc)

    return delta.clone().detach().requires_grad_(False)


def poison_p3(args, loader, model, writer):
    # Generate universal perturbations for each class
    poisons = []
    for i in range(args.num_classes):
        poison = universal_target_attack(model, loader, i, writer, args)
        poisons.append(poison.squeeze())
    poisons = torch.stack(poisons)
    poisons = poisons[list(range(1, 10))+[0]]   # using a fixed permutation of labels

    vis = make_grid(poisons, nrow=5, normalize=True, scale_each=True)
    writer.add_image('universal_perturbation', vis, global_step=1)

    poisoned_data = poison_p3p4p5(args, loader, model, writer, poisons)
    return poisoned_data


def poison_p4(args, loader, model, writer):
    # Generate universal perturbations for each class
    poisons = []
    for i in range(args.num_classes):
        poison = universal_target_attack(model, loader, i, writer, args)
        poisons.append(poison.squeeze())
    poisons = torch.stack(poisons)

    vis = make_grid(poisons, nrow=5, normalize=True, scale_each=True)
    writer.add_image('universal_perturbation', vis, global_step=0)

    poisoned_data = poison_p3p4p5(args, loader, model, writer, poisons)
    return poisoned_data


def poison_p5(args, loader, model, writer):
    # Generate random perturbations for each class
    poisons = torch.zeros(args.num_classes, *args.data_shape).cuda(non_blocking=True)

    step = STEPS[args.constraint](None, args.eps, None)
    poisons = step.random_perturb(poisons)

    vis = make_grid(poisons, nrow=5, normalize=True, scale_each=True)
    writer.add_image('random_perturbation', vis, global_step=0, dataformats='CHW')

    poisoned_data = poison_p3p4p5(args, loader, model, writer, poisons)
    return poisoned_data



def poisoning(args, loader, model, writer):
    set_seed(args.seed)

    if args.poison_type == 'P1':
        poisoned_data = poison_p1(args, loader, model, writer)
    elif args.poison_type == 'P2':
        poisoned_data = poison_p2(args, loader, model, writer)
    elif args.poison_type == 'P3':
        poisoned_data = poison_p3(args, loader, model, writer)
    elif args.poison_type == 'P4':
        poisoned_data = poison_p4(args, loader, model, writer)
    elif args.poison_type == 'P5':
        poisoned_data = poison_p5(args, loader, model, writer)
        
    torch.save(poisoned_data, args.poison_file_path)


def visualization(args, writer):
    clean_set = datasets.CIFAR10(args.clean_data_path, train=True, transform=transform_test)
    poison_set = CIFAR10Poisoned(args.poison_data_path, args.constraint, args.poison_type, transform=transform_test)

    clean_loader = DataLoader(clean_set, batch_size=5, shuffle=False, num_workers=8)
    poison_loader = DataLoader(poison_set, batch_size=5, shuffle=False, num_workers=8)

    clean_iterator = iter(clean_loader)
    poison_iterator = iter(poison_loader)
    for i in range(3):
        clean_inp, label = next(clean_iterator)
        poison_inp, label = next(poison_iterator)

        imgs = torch.cat([clean_inp, poison_inp], dim=0)
        vis = make_grid(imgs, nrow=5, normalize=False, scale_each=False)
        writer.add_image('poisoned_examples', vis, global_step=i, dataformats='CHW')

        ylist = None
        # ylist = ['$\mathcal{D}$', '$\widehat{\mathcal{D}}_{\mathsf{P5}}$']
        # clean_set.classes[1] = 'car'

        show_image_row([clean_inp, poison_inp],
                ylist=ylist,
                tlist=[[clean_set.classes[int(t)] for t in l] for l in [label, label]],
                fontsize=20,
                filename=os.path.join(os.path.join(args.out_dir, args.exp_name), 'poisoned_examples_{}.png'.format(i)))

# def main(args):
#     visualization(args, None)

def main(args):

    if os.path.isfile(args.poison_file_path):
        print('Poison [{}] already exists.'.format(args.poison_file_path))
        return
    
    data_set = datasets.CIFAR10(args.clean_data_path, train=True, download=True, transform=transform_test)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    model = make_and_restore_model(args.arch, resume_path=args.model_path)
    model.eval()
    writer = SummaryWriter(args.tensorboard_path)

    poisoning(args, data_loader, model, writer)
    
    visualization(args, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate poisoned dataset for CIFAR10')
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=0.5, type=float)
    parser.add_argument('--constraint', default='L2', choices=['Linf', 'L2'], type=str)

    parser.add_argument('--arch', default='VGG16', type=str, choices=['VGG16', ])
    parser.add_argument('--model_path', default='results/VGG16-STonC-lr0.1-bs128-wd0.0005-seed0/checkpoint.pth', type=str)

    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--out_dir', default='results/', type=str)
    parser.add_argument('--clean_data_path', default='./datasets/CIFAR10', type=str)
    parser.add_argument('--poison_data_path', default='./datasets/CIFAR10Poison', type=str)
    parser.add_argument('--poison_type', default='C', choices=['P1', 'P2', 'P3', 'P4', 'P5'])

    parser.add_argument('--gpuid', default=0, type=int)

    args = parser.parse_args()

    args.exp_name = '{}-{}-{}-eps{:.5f}'.format(args.arch, args.poison_type, args.constraint, args.eps)
    args.tensorboard_path = os.path.join(os.path.join(args.out_dir, args.exp_name), 'tensorboard')
    args.batch_size = 256
    args.num_classes = 10
    args.data_shape = (3, 32, 32)
    args.num_steps = 100
    args.step_size = args.eps / 5

    args.poison_data_path = os.path.expanduser(args.poison_data_path)
    if not os.path.exists(args.poison_data_path):
        os.makedirs(args.poison_data_path)
    args.poison_file_path = os.path.join(args.poison_data_path, '{}.{}'.format(args.constraint, args.poison_type.lower()))

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
