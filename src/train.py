import argparse
import os
import random
import shutil
import time
import warnings
import utils
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
import numpy as np
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from apex import parallel
from apex.parallel.LARC import LARC

from model import Model
from loss import Loss
import vit as vits

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))
model_names = ['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'] + torchvision_archs

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
parser.add_argument('--final-lr', default=None, type=float,
                    help='final learning rate (None for constant learning rate)')
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
parser.add_argument('-p', '--print-freq', default=16, type=int,
                    metavar='N', help='print frequency (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cls-size', type=int, default=[1000], nargs='+',
                    help='size of classification layer. can be a list if cls-size > 1')
parser.add_argument('--num-cls', default=1, type=int, metavar='NCLS',
                    help='number of classification layers')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--rm-pretrained-cls', action='store_true',
                    help='ignore classifier when loading pretrained model (used for initializing imagenet subset)')
parser.add_argument('--queue-len', default=262144, type=int,
                    help='length of nearest neighbor queue')
parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=4096, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=3, type=int,
                    help='number of MLP hidden layers')
parser.add_argument('--row-tau', default=0.1, type=float,
                    help='row softmax temperature (default: 0.1)')
parser.add_argument('--col-tau', default=0.05, type=float,
                    help='column softmax temperature (default: 0.05)')
parser.add_argument('--use-amp', action='store_true',
                    help='use automatic mixed precision')
parser.add_argument("--syncbn_process_group_size", default=0, type=int,
                    help='process group size for syncBN layer')
parser.add_argument('--use-lsf-env', action='store_true',
                    help='use LSF env variables')
parser.add_argument('--use-bn', action='store_true',
                    help='use batch normalization layers in MLP')
parser.add_argument('--fixed-cls', action='store_true',
                    help='use a fixed classifier')
parser.add_argument('--global-crops-scale', type=float, nargs='+', default=(0.4, 1.),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
                    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we 
                    recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
parser.add_argument('--local-crops-number', type=int, default=6,
                    help="""Number of small local views to generate. 
                    Set this parameter to 0 to disable multi-crop training. 
                    When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
parser.add_argument('--local-crops-scale', type=float, nargs='+', default=(0.05, 0.4),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image. 
                    Used for small local view cropping of multi-crop.""")
parser.add_argument('--patch-size', default=16, type=int,
                    help="""Size in pixels of input square patches - default 16 (for 16x16 patches). Using smaller 
                    values leads to better performance but requires more memory. 
                    Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling 
                    mixed precision training to avoid unstabilities.""")
parser.add_argument('--clip-grad', type=float, default=0.0,
                    help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can 
                    help optimization for larger ViT architectures. 0 for disabling.""")
parser.add_argument('--no-nn-aug', action='store_true',
                    help='do not use nearest neighbor augmentation')
parser.add_argument('--no-bias-wd', action='store_true',
                    help='do not regularize biases nor Norm parameters')
parser.add_argument('--bbone-wd', type=float, default=None,
                    help='backbone weight decay. if set to None weight_decay is used for backbone as well.')
parser.add_argument('--eps', type=float, default=1e-12,
                    help='small value to avoid division by zero and log(0)')
parser.add_argument('--subset-file', default=None, type=str,
                    help='path to imagenet subset txt file')
parser.add_argument('--no-leaky', action='store_true',
                    help='use regular relu layers instead of leaky relu in MLP')


def main():
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # Slurm
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_TASKS_PER_NODE"][0])
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

    # save log only for master process
    if args.rank == 0 or not args.distributed:
        sys.stdout = utils.PrintMultiple(sys.stdout, open(os.path.join(args.save_path, 'log.txt'), 'a+'))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch in vits.__dict__.keys():
        base_model = vits.__dict__[args.arch](patch_size=args.patch_size)
        backbone_dim = base_model.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        base_model = torchvision_models.__dict__[args.arch]()
        backbone_dim = base_model.fc.weight.shape[1]
    else:
        raise Exception("Unknown architecture: {}".format(args.arch))
    model = Model(base_model=base_model,
                  dim=args.dim,
                  hidden_dim=args.hidden_dim,
                  cls_size=args.cls_size,
                  num_cls=args.num_cls,
                  num_hidden=args.num_hidden,
                  use_bn=args.use_bn,
                  backbone_dim=backbone_dim,
                  fixed_cls=args.fixed_cls,
                  no_leaky=args.no_leaky)
    if args.distributed:
        process_group = parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=process_group)
    print(model)

    # nearest neighbor queue
    nn_queue = utils.NNQueue(args.queue_len, args.dim, args.gpu)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # load state dictionary
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                # remove classifier if necessary
                if args.rm_pretrained_cls and 'cls_' in k:
                    del state_dict[k]

                # remove module. prefix
                elif k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert len(msg.missing_keys) == 0, "missing_keys: {}".format(msg.missing_keys)
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
    criterion = Loss(row_tau=args.row_tau, col_tau=args.col_tau, eps=args.eps).cuda(args.gpu)
    params_groups = utils.get_params_groups(model, args)
    if args.sgd:
        optimizer = torch.optim.SGD(params_groups, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params_groups, args.lr,
                                      weight_decay=args.weight_decay)

    if args.lars:
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    # optionally resume from a checkpoint
    last_model_path = os.path.join(args.save_path, 'model_last.pth.tar')
    if not args.resume and os.path.isfile(last_model_path):  # automatic resume
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
            nn_queue = checkpoint['nn_queue']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data loading code
    traindir = os.path.join(args.data, 'train')
    transform = utils.DataAugmentation(args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    dataset = utils.ImageFolderWithIndices(traindir, transform=transform)
    if args.subset_file is not None:
        dataset = utils.imagenet_subset(dataset, args.subset_file)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                                         num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True)

    # schedulers
    lr_schedule = utils.cosine_scheduler_with_warmup(base_value=args.lr,
                                                     final_value=args.final_lr,
                                                     epochs=args.epochs,
                                                     niter_per_ep=len(loader),
                                                     warmup_epochs=args.warmup_epochs,
                                                     start_warmup_value=args.start_warmup)

    # mixed precision
    scaler = GradScaler(enabled=args.use_amp, init_scale=2. ** 14)

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        # train for one epoch
        loss_i, acc1 = train(loader, model, nn_queue, scaler, criterion, optimizer, lr_schedule, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = True if epoch == 0 else loss_i < best_loss
        best_loss = loss_i if epoch == 0 else min(loss_i, best_loss)

        if not args.distributed or (args.distributed and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'nn_queue': nn_queue,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, is_milestone=(epoch + 1) % 25 == 0,
                filename=os.path.join(args.save_path, 'model_last.pth.tar'))


def train(loader, model, nn_queue, scaler, criterion, optimizer, lr_schedule, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.6f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets, indices) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cos:
            # update learning rate
            adjust_lr(optimizer, lr_schedule, iteration=epoch * len(loader) + i)

        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = [x.cuda(args.gpu, non_blocking=True) for x in images]
            targets = targets.cuda(args.gpu, non_blocking=True)  # only used for monitoring progress, NOT for training
            indices = indices.cuda(args.gpu, non_blocking=True)

        with autocast(enabled=args.use_amp):
            # compute embeddings
            embds = model(images, return_embds=True)

            # view1 embeddings
            embds1 = embds[0].clone().detach()

            if nn_queue.full:
                if not args.no_nn_aug:  # if queue is full and nn is enabled, replace view1 with view1-nn
                    embds[0], nn_targets = nn_queue.get_nn(embds1, indices)
                else:  # if nn augmentation is disabled do not replace, but use for monitoring progress
                    _, nn_targets = nn_queue.get_nn(embds1, indices)

                # measure accuracy of nearest neighbor (for monitoring progress)
                acc1 = (targets.view(-1, ) == nn_targets.view(-1, )).float().mean().view(1, ) * 100.0
                # compute accuracy of all workers
                acc1 = utils.AllGather.apply(acc1).mean()
                top1.update(acc1, targets.size(0))

            # gather embeddings, targets and indices from all workers
            embds1 = utils.AllGather.apply(embds1)
            targets = utils.AllGather.apply(targets)
            indices = utils.AllGather.apply(indices)

            # push embeddings of view1 (all workers) into queue
            nn_queue.push(embds1, targets, indices)

            # compute probs
            probs = model(embds, return_embds=False)

            with autocast(enabled=False):
                # compute loss
                loss = criterion(probs)

        assert not torch.isnan(loss), 'loss is nan!'

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        if args.clip_grad:
            scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            _ = utils.clip_gradients(model, args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        # record loss
        loss = loss.detach() / dist.get_world_size()
        dist.all_reduce(loss)  # compute mean over all workers
        losses.update(loss.item(), probs[0][0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == args.print_freq - 1:
            # debug
            target = probs[0][1].clone().detach().argmax(dim=1)
            unique_predictions = torch.unique(utils.AllGather.apply(target)).shape[0]
            print('number of unique predictions (cls {}): {}'.format(0, unique_predictions))
            progress.display(i)

    return losses.avg, top1.avg


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


def adjust_lr(optimizer, lr_schedule, iteration):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_schedule[iteration]


if __name__ == '__main__':
    main()
