import argparse
import os
import random
import shutil
import time
import sys
import functools
import numpy as np
import utils
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

from model import Model
from resnet_cifar import resnet18
from loss import Loss
from PIL import Image
from apex.parallel.LARC import LARC

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as adjusted_nmi
from sklearn.metrics import adjusted_rand_score as adjusted_rand_index


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser(description='PyTorch Self-Supervised Classification of CIFAR')
parser.add_argument('--data', metavar='DATA', default='CIFAR10',
                    choices=['CIFAR10', 'CIFAR100-20'],
                    help='training dataset (default: imagenet)')
parser.add_argument('--data_path', default='../small_datasets/', type=str,
                    help='directory for saving datasets')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.8, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--final-lr', default=0.008, type=float,
                    help='final learning rate')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--sgd', action='store_true',
                    help='use SGD optimizer')
parser.add_argument('--lars', action='store_true',
                    help='use LARS optimizer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--cls-size', default=30, type=int, metavar='CLS',
                    help='size of classification layer')
parser.add_argument('--num-cls', default=30, type=int, metavar='NCLS',
                    help='number of classification layers')
parser.add_argument('--save-path', default='../cifar10_exp/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--dim', default=64, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=256, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=2, type=int,
                    help='number of MLP hidden layers (1, 2 or 3 only)')
parser.add_argument('--tau', default=0.3, type=float,
                    help='softmax temperature (default: 0.3)')
parser.add_argument('--no-mlp', action='store_true',
                    help='do not use MLP')


def main():
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create output directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)

    # # save log
    sys.stdout = utils.PrintMultiple(sys.stdout, open(os.path.join(args.save_path, 'log.txt'), 'a+'))
    print(args)

    # create model
    print("=> creating model '{}'".format('ResNet18'))
    model = Model(base_model=resnet18,
                  dim=args.dim,
                  hidden_dim=args.hidden_dim,
                  cls_size=args.cls_size,
                  tau=args.tau,
                  num_cls=args.num_cls,
                  no_mlp=args.no_mlp,
                  num_hidden=args.num_hidden,
                  backbone_dim=512)
    model = torch.nn.DataParallel(model).cuda()

    # load from pre-trained
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # load state dictionary
            state_dict = checkpoint['state_dict']

            args.start_epoch = 0
            model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # define loss function (criterion) and optimizer
    criterion = Loss(normalize_rows=True).cuda()
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)

    if args.lars:
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    # optionally resume from a checkpoint
    last_model_path = os.path.join(args.save_path, 'model_last.pth.tar')
    if not args.resume and os.path.isfile(last_model_path):
        args.resume = last_model_path
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.data == 'CIFAR100-20':
        args.data = 'CIFAR100'
        target_transform = _cifar100_to_cifar20
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
    else:
        target_transform = None
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.247, 0.243, 0.261])

    # augmentations
    train_augmentations = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_augmentations = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # datasets
    class PairDataset(getattr(datasets, args.data)):
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                view1 = self.transform(img)
                view2 = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return view1, view2, target

    train_dataset = PairDataset(root=args.data_path, train=True,
                                transform=train_augmentations,
                                target_transform=target_transform,
                                download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)

    val_dataset = getattr(datasets, args.data)(root=args.data_path, train=False,
                                               transform=val_augmentations,
                                               target_transform=target_transform,
                                               download=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.cos:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss_i, acc1 = train(train_loader, model, criterion, optimizer, epoch, args)

        # remember best loss value and save checkpoint
        is_best = True if epoch == 0 else loss_i < best_loss
        best_loss = loss_i if epoch == 0 else min(loss_i, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, is_milestone=(epoch + 1) % 50 == 0,
            filename=os.path.join(args.save_path, 'model_last.pth.tar'))

        # validate
        validate(val_loader, model, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.6f')
    top5 = AverageMeter('Acc@5', ':6.6f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (batch_view1, batch_view2, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            batch_view1 = batch_view1.cuda()
            batch_view2 = batch_view2.cuda()

        out1, out2 = model(view1=batch_view1, view2=batch_view2)
        loss = criterion(out1, out2)

        # measure accuracy and record loss
        target = out2[0].clone().detach().argmax(dim=1)
        acc1, acc5 = accuracy(out1[0], target, topk=(1, 5))
        losses.update(loss.item(), out2[0].size(0))
        top1.update(acc1[0], out2[0].size(0))
        top5.update(acc5[0], out2[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == args.print_freq - 1:
            progress.display(i)

    return losses.avg, top1.avg


def validate(val_loader, model, args):
    all_preds = np.zeros((len(val_loader.dataset.targets), ), dtype=int)
    all_targets = np.zeros((len(val_loader.dataset.targets), ), dtype=int)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()

            # compute output
            output = model(images)

            num_samples = output.shape[0]

            # save target
            all_targets[idx: idx + num_samples] = targets.numpy()

            # compute prediction
            all_preds[idx: idx + num_samples] = output.clone().detach().argmax(dim=1).cpu().numpy()

            idx += num_samples

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    # compute various measurements
    val_nmi = nmi(all_targets, all_preds)
    val_adjusted_nmi = adjusted_nmi(all_targets, all_preds)
    val_adjusted_rand_index = adjusted_rand_index(all_targets, all_preds)
    print('=> number of samples: {}'.format(len(all_targets)))
    print('=> NMI: {:.3f}%'.format(val_nmi * 100.0))
    print('=> Adjusted NMI: {:.3f}%'.format(val_adjusted_nmi * 100.0))
    print('=> Adjusted Rand-Index: {:.3f}%'.format(val_adjusted_rand_index * 100.0))

    return


def save_checkpoint(state, is_best, is_milestone, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))
        print('Best model was saved.')
    if is_milestone:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_{}.pth.tar'.format(state['epoch'])))
        print('Milestone {} model was saved.'.format(state['epoch']))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1, ).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on cosine schedule"""
    lr = args.final_lr + (args.lr - args.final_lr) * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    print('Epoch: {}, lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _cifar100_to_cifar20(target):
    # from IIC github
    _dict = \
    {0: 4,
     1: 1,
     2: 14,
     3: 8,
     4: 0,
     5: 6,
     6: 7,
     7: 7,
     8: 18,
     9: 3,
     10: 3,
     11: 14,
     12: 9,
     13: 18,
     14: 7,
     15: 11,
     16: 3,
     17: 9,
     18: 7,
     19: 11,
     20: 6,
     21: 11,
     22: 5,
     23: 10,
     24: 7,
     25: 6,
     26: 13,
     27: 15,
     28: 3,
     29: 15,
     30: 0,
     31: 11,
     32: 1,
     33: 10,
     34: 12,
     35: 14,
     36: 16,
     37: 9,
     38: 11,
     39: 5,
     40: 5,
     41: 19,
     42: 8,
     43: 8,
     44: 15,
     45: 13,
     46: 14,
     47: 17,
     48: 18,
     49: 10,
     50: 16,
     51: 4,
     52: 17,
     53: 4,
     54: 2,
     55: 0,
     56: 17,
     57: 4,
     58: 18,
     59: 17,
     60: 10,
     61: 3,
     62: 2,
     63: 12,
     64: 12,
     65: 16,
     66: 12,
     67: 1,
     68: 9,
     69: 19,
     70: 2,
     71: 10,
     72: 0,
     73: 1,
     74: 16,
     75: 12,
     76: 9,
     77: 13,
     78: 15,
     79: 13,
     80: 16,
     81: 19,
     82: 2,
     83: 4,
     84: 6,
     85: 19,
     86: 5,
     87: 5,
     88: 8,
     89: 19,
     90: 18,
     91: 1,
     92: 2,
     93: 15,
     94: 6,
     95: 0,
     96: 17,
     97: 8,
     98: 14,
     99: 13}

    return _dict[target]


if __name__ == '__main__':
    main()
