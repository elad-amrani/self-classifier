import argparse
import os
import random
import shutil
import time
import warnings
import utils
import sys
import math
import functools
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from apex import parallel
from apex.parallel.LARC import LARC

from model import Model
from loss import Loss

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as adjusted_nmi
from sklearn.metrics import adjusted_rand_score as adjusted_rand_index


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Self-Supervised Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=10, type=int,
                    help='linear warmup epochs (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=4.8, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--start-warmup', default=0.3, type=float,
                    help='initial warmup learning rate')
parser.add_argument('--final-lr', default=0.0048, type=float,
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate self-supervised classification on the validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cls-size', default=3000, type=int, metavar='CLS',
                    help='size of classification layer')
parser.add_argument('--num-cls', default=10, type=int, metavar='NCLS',
                    help='number of classification layers')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--use-aug', action='store_true',
                    help='Use special augmentations (color jitter, grayscale and Gaussian blur)')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=2048, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=1, type=int,
                    help='number of MLP hidden layers (1 or 2 only)')
parser.add_argument('--subset-size', default=-1, type=int,
                    help='train on only subset-size number of samples. set -1 for entire dataset')
parser.add_argument('--tau', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')
parser.add_argument('--use-amp', action='store_true',
                    help='use automatic mixed precision')
parser.add_argument("--syncbn_process_group_size", default=0, type=int,
                    help='process group size for syncBN layer')
parser.add_argument('--use-lsf-env', action='store_true',
                    help='use LSF env variables')
parser.add_argument('--no-mlp', action='store_true',
                    help='do not use MLP')
parser.add_argument('--learnable-cls', action='store_true',
                    help='require gradient for classifier (default: fixed orthogonal classifiers)')

def main():
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # Slurm
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
        args.gpu = args.rank % torch.cuda.device_count()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.distributed:
        if args.use_lsf_env:
            args.rank, args.gpu = utils.set_lsf_env(world_size=args.world_size)
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        else:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    rank=args.rank, world_size=args.world_size)

    # save log only for rank 0
    if args.rank == 0 or not args.distributed:
        sys.stdout = utils.PrintMultiple(sys.stdout, open(os.path.join(args.save_path, 'log.txt'), 'a+'))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = Model(base_model=models.__dict__[args.arch],
                  dim=args.dim,
                  hidden_dim=args.hidden_dim,
                  cls_size=args.cls_size,
                  tau=args.tau,
                  num_cls=args.num_cls,
                  no_mlp=args.no_mlp,
                  num_hidden=args.num_hidden,
                  learnable_cls=args.learnable_cls)
    if args.distributed:
        process_group = parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = parallel.convert_syncbn_model(model, process_group=process_group)
    print(model)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # load state dictionary
            state_dict = checkpoint['state_dict']

            # remove module. prefix
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]

            args.start_epoch = 0
            model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = Loss().cuda(args.gpu)
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
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
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

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.use_aug:
        print('=> using special augmentations.')
        train_augmentations = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_augmentations = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = utils.SscDataset(root_dir=traindir,
                                     transform=train_augmentations,
                                     size=args.subset_size)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, args)
        return

    # scheduler (taken from SwAV)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    args.final_lr = args.final_lr if args.cos else args.lr
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.lr - args.final_lr) *
                                   (1 + math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs))))
                                   for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # tensorboard writer
    if args.rank == 0 or not args.distributed:
        writer = SummaryWriter(log_dir=args.save_path)

    scaler = GradScaler(enabled=args.use_amp)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        loss_i, acc1 = train(train_loader, model, scaler, criterion, optimizer, lr_schedule, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = True if epoch == 0 else loss_i < best_loss
        best_loss = loss_i if epoch == 0 else min(loss_i, best_loss)

        if not args.distributed or (args.distributed and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, is_milestone=(epoch + 1) % 50 == 0,
                filename=os.path.join(args.save_path, 'model_last.pth.tar'))

            # write tensorboard
            writer.add_scalar('Loss', loss_i, epoch)
            writer.add_scalar('Top1 Accuracy', acc1, epoch)


def train(train_loader, model, scaler, criterion, optimizer, lr_schedule, epoch, args):
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

        # update learning rate
        adjust_learning_rate(optimizer, lr_schedule, iteration=epoch * len(train_loader) + i)

        optimizer.zero_grad()

        if args.gpu is not None:
            batch_view1 = batch_view1.cuda(args.gpu, non_blocking=True)
            batch_view2 = batch_view2.cuda(args.gpu, non_blocking=True)

        with autocast(enabled=args.use_amp):
            out1, out2 = model(view1=batch_view1, view2=batch_view2)
            loss = criterion(out1, out2)

        # measure accuracy and record loss
        target = out2[0].clone().detach().argmax(dim=1)
        acc1, acc5 = accuracy(out1[0], target, topk=(1, 5))
        losses.update(loss.item(), out2[0].size(0))
        top1.update(acc1[0], out2[0].size(0))
        top5.update(acc5[0], out2[0].size(0))

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

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


def adjust_learning_rate(optimizer, lr_schedule, iteration):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_schedule[iteration]


if __name__ == '__main__':
    main()
