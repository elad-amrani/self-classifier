import argparse
import os
import random
import time
import warnings
import sys
import functools
import numpy as np
import math
import utils
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn.functional as F

from model import Model

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
parser.add_argument('--cls-size', default=3000, type=int, metavar='CLS',
                    help='size of classification layer')
parser.add_argument('--num-cls', default=10, type=int, metavar='NCLS',
                    help='number of classification layers')
parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=2048, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=1, type=int,
                    help='number of MLP hidden layers (1 or 2 only)')
parser.add_argument('--tau', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')
parser.add_argument('--no-mlp', action='store_true',
                    help='do not use MLP')
parser.add_argument('--num-classes', default=15, type=int,
                    help='number of low entropy classes to visualize')
parser.add_argument('--num-samples-per-class', default=9, type=int,
                    help='number of samples per class to visualize (must be a square number)')

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
    model = Model(base_model=models.__dict__[args.arch],
                  dim=args.dim,
                  hidden_dim=args.hidden_dim,
                  cls_size=args.cls_size,
                  tau=args.tau,
                  num_cls=args.num_cls,
                  no_mlp=args.no_mlp,
                  num_hidden=args.num_hidden)
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
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # run inference
    targets, preds, class_entropy, class_num_samples, class_label = validate(val_loader, model, args)

    # compute metrics
    compute_metrics(targets, preds)

    # extract num_classes low entropy pseudo classes
    low_entropy_classes = class_label[:args.num_classes]

    # sample num_samples_per_class from low entropy classes and save grid image
    for idx_label_i, label_i in enumerate(low_entropy_classes):
        # extract all indices of current class
        sample_indices = np.where(preds == label_i)[0]

        # sample randomly num_samples_per_class
        np.random.seed(0)
        subset_sample_indices = np.random.choice(sample_indices, args.num_samples_per_class, replace=False)

        # get image paths
        subset_img_paths = [x[0] for idx_x, x in enumerate(val_loader.dataset.imgs) if idx_x in subset_sample_indices]

        # get images
        subset_images = [load_image(x) for x in subset_img_paths]

        # get grid
        grid_i = image_grid(subset_images, int(math.sqrt(args.num_samples_per_class)), int(math.sqrt(args.num_samples_per_class)))

        # save grid
        grid_i.save(os.path.join(args.save_path, 'grid_{}.pdf'.format(idx_label_i)))
        print('=> saved grid_{}.pdf'.format(idx_label_i))


def validate(val_loader, model, args):
    all_preds = np.zeros((len(val_loader.dataset.targets), ), dtype=int)
    all_targets = np.zeros((len(val_loader.dataset.targets), ), dtype=int)
    class_entropy = np.zeros((args.cls_size,), dtype=float)
    class_num_samples = np.zeros((args.cls_size,), dtype=int)
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
            output = F.log_softmax(model(images), dim=1).detach().cpu().numpy()

            num_samples = output.shape[0]

            # save target
            all_targets[idx: idx + num_samples] = targets.numpy()

            # compute prediction
            preds_i = output.argmax(1)
            all_preds[idx: idx + num_samples] = preds_i

            # compute class entropy
            entropy_i = - output.sum(1)
            for idx_pred, pred in enumerate(preds_i):
                class_entropy[pred] += entropy_i[idx_pred]
                class_num_samples[pred] += 1

            idx += num_samples

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    # extract only classes with more than or equal to num_samples_per_class
    class_label = np.arange(args.cls_size)[np.where(class_num_samples >= args.num_samples_per_class)[0]]
    class_entropy = class_entropy[np.where(class_num_samples >= args.num_samples_per_class)[0]]
    class_num_samples = class_num_samples[np.where(class_num_samples >= args.num_samples_per_class)[0]]

    # average class entropy
    class_entropy = class_entropy / class_num_samples

    # sort by entropy
    sorted_indices = np.argsort(class_entropy)
    class_label = class_label[sorted_indices]
    class_entropy = class_entropy[sorted_indices]
    class_num_samples = class_num_samples[sorted_indices]

    return all_targets, all_preds, class_entropy, class_num_samples, class_label


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


def compute_metrics(targets, preds):
    val_nmi = nmi(targets, preds)
    val_adjusted_nmi = adjusted_nmi(targets, preds)
    val_adjusted_rand_index = adjusted_rand_index(targets, preds)
    print('=> number of samples: {}'.format(len(targets)))
    print('=> number of unique assignments: {}'.format(len(set(preds))))
    print('=> NMI: {:.3f}%'.format(val_nmi * 100.0))
    print('=> Adjusted NMI: {:.3f}%'.format(val_adjusted_nmi * 100.0))
    print('=> Adjusted Rand-Index: {:.3f}%'.format(val_adjusted_rand_index * 100.0))


def load_image(infilename):
    img = Image.open(infilename)
    img = img.resize((128, 128))
    return img


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == '__main__':
    main()
