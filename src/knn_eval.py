import argparse
import os
import random
import time
import warnings
import utils
import sys
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.neighbors import KNeighborsClassifier


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser(description='K-NN Evaluation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--subset-size', default=-1, type=int,
                    help='train on only subset-size number of samples. set -1 for entire dataset')
parser.add_argument('--kk', default=20, type=int,
                    help='k parameter of k-nn classifier (default: 20)')


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

    # save log file
    sys.stdout = utils.PrintMultiple(sys.stdout, open(os.path.join(args.save_path, 'log.txt'), 'a+'))
    print(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    model.fc = nn.Identity()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # load state dictionary
            state_dict = checkpoint['state_dict']

            # remove module. prefix
            for k in list(state_dict.keys()):
                if k.startswith('module.backbone.'):
                    # remove prefix
                    state_dict[k[len("module.backbone."):]] = state_dict[k]
                    del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert len(msg.missing_keys) == 0
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            return

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        print('=> using {} GPUs.'.format(torch.cuda.device_count()))
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = utils.SscDataset(root_dir=traindir,
                                     transform=transform,
                                     size=args.subset_size)

    val_dataset = utils.SscDataset(root_dir=valdir,
                                   transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # train inference
    train_features, train_labels = inference(train_loader, model, args, prefix='Train Set Inference: ')

    # val inference
    val_features, val_labels = inference(val_loader, model, args, prefix='Test Set Inference: ')

    # compute knn accuracy
    knn_accuracy(train_features, train_labels, val_features, val_labels, args.kk)


def inference(loader, model, args, prefix):
    all_features = np.zeros((len(loader.dataset), 2048), dtype=np.float)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix=prefix)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, _, indices) in enumerate(loader):
            if torch.cuda.is_available():
                images = images.cuda()

            # compute output
            output = model(images)

            # compute prediction
            all_features[indices] = F.normalize(output, dim=1).detach().cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    # extract labels from img name
    all_labels = np.array([os.path.split(os.path.split(x)[0])[1] for x in loader.dataset.img_paths])

    return all_features, all_labels


def knn_accuracy(train_features, train_labels, val_features, val_labels, kk):
    knn = KNeighborsClassifier(n_neighbors=kk, weights='distance', n_jobs=-1).fit(train_features, train_labels)
    acc = knn.score(val_features, val_labels)

    print('{}-Nearest neighbor accuracy: {:.3f}'.format(kk, acc))


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


if __name__ == '__main__':
    main()
