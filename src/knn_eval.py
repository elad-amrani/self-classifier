import argparse
import os
import random
import time
import warnings
import utils
import sys
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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

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
parser.add_argument('-b', '--train-batch-size', default=256, type=int,
                    help='train set batch size')
parser.add_argument('--val-batch-size', default=256, type=int,
                    help='validation set batch size')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--nb-knn', default=[10, 20, 100, 200], nargs='+', type=int,
                    help='Number of NN to use')
parser.add_argument('--label-subset', default="100", type=str, choices=["1", "10", "100"],
                    help='percentage of labeled data: 1%, 10% or 100% (default: 1)')
parser.add_argument('--tau', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')
parser.add_argument('--num-classes', default=1000, type=int,
                    help='number of classes (1000 for ImageNet, 10 (default: 0.1)')
parser.add_argument('--cls-size', type=int, default=[1000], nargs='+',
                    help='size of classification layer. can be a list if cls-size > 1')
parser.add_argument('--num-cls', default=1, type=int, metavar='NCLS',
                    help='number of classification layers')
parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=4096, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=3, type=int,
                    help='number of MLP hidden layers')
parser.add_argument('--use-bn', action='store_true',
                    help='use batch normalization layers in MLP')
parser.add_argument('--use-mlp', action='store_true',
                    help='use features after MLP. By default uses features from backbone')
parser.add_argument('--load-features', action='store_true',
                    help='use features from earlier dump (in args.save_path)')
parser.add_argument('--use-half', action='store_true',
                    help='use half precision for inference (in case not enough GPU memory)')
parser.add_argument('--subset-file', default=None, type=str,
                    help='path to imagenet subset txt file')
parser.add_argument('--no-leaky', action='store_true',
                    help='use regular relu layers instead of leaky relu in MLP')


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
    args.backbone_dim = model.fc.weight.shape[1]
    if args.use_mlp:
        if args.num_hidden == 1:
            model.fc = nn.Linear(args.backbone_dim, args.dim)
        else:
            layers = [nn.Linear(args.backbone_dim, args.hidden_dim)]
            if args.use_bn:
                layers.append(nn.BatchNorm1d(args.hidden_dim))
            if args.no_leaky:
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.LeakyReLU(inplace=True))
            for _ in range(args.num_hidden - 2):
                layers.append(nn.Linear(args.hidden_dim, args.hidden_dim))
                if args.use_bn:
                    layers.append(nn.BatchNorm1d(args.hidden_dim))
                if args.no_leaky:
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Linear(args.hidden_dim, args.dim))
            model.fc = nn.Sequential(*layers)
    else:
        model.fc = nn.Identity()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # load state dictionary
            state_dict = checkpoint['state_dict']

            # remove module.backbone prefix
            for k in list(state_dict.keys()):
                if k.startswith('module.backbone.'):
                    # remove prefix
                    state_dict[k[len("module.backbone."):]] = state_dict[k]
                    del state_dict[k]

            if args.use_mlp:
                # replace module.mls_head.mlp prefix with fc
                for k in list(state_dict.keys()):
                    if k.startswith('module.mlp_head.mlp.'):
                        # replace prefix
                        state_dict['fc.' + k[len("module.mlp_head.mlp."):]] = state_dict[k]
                        del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert len(msg.missing_keys) == 0
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
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

    train_dataset = utils.ImageFolderWithIndices(traindir, transform)
    if args.label_subset != "100":
        train_dataset = utils.imagenet_subset_samples(train_dataset, traindir, args.label_subset)  # extract subset
    val_dataset = utils.ImageFolderWithIndices(valdir, transform)

    if args.subset_file is not None:
        train_dataset = utils.imagenet_subset(train_dataset, args.subset_file)
        val_dataset = utils.imagenet_subset(val_dataset, args.subset_file)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.load_features:
        train_features = torch.load(os.path.join(args.save_path, "trainfeat.pth"))
        val_features = torch.load(os.path.join(args.save_path, "valfeat.pth"))
        train_labels = torch.load(os.path.join(args.save_path, "trainlabels.pth"))
        val_labels = torch.load(os.path.join(args.save_path, "vallabels.pth"))
    else:
        # train inference
        train_features, train_labels = inference(train_loader, model, args, prefix='Train Set Inference: ')
        # val inference
        val_features, val_labels = inference(val_loader, model, args, prefix='Test Set Inference: ')

        # dump
        torch.save(train_features.cpu(), os.path.join(args.save_path, "trainfeat.pth"))
        torch.save(val_features.cpu(), os.path.join(args.save_path, "valfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.save_path, "trainlabels.pth"))
        torch.save(val_labels.cpu(), os.path.join(args.save_path, "vallabels.pth"))

    # compute knn accuracy
    print("Features are ready!\nStart the k-NN classification.")
    if args.use_half:
        train_features = train_features.half()
        val_features = val_features.half()

    if torch.cuda.is_available():
        train_features = train_features.cuda(non_blocking=True)
        val_features = val_features.cuda(non_blocking=True)
        train_labels = train_labels.cuda(non_blocking=True)
        val_labels = val_labels.cuda(non_blocking=True)

    for kk in args.nb_knn:
        top1, top5 = knn_classifier(train_features, train_labels, val_features, val_labels, kk, args)
        print(f"{kk}-NN classifier result: Top1: {top1}, Top5: {top5}")


@torch.no_grad()
def inference(loader, model, args, prefix):
    out_dim = args.dim if args.use_mlp else args.backbone_dim
    all_features = np.zeros((len(loader.dataset), out_dim), dtype=np.float)
    all_labels = np.zeros((len(loader.dataset), ), dtype=np.int)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix=prefix)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, targets, indices) in enumerate(loader):
        if torch.cuda.is_available():
            images = images.cuda()

        # compute output
        output = model(images)

        # compute prediction
        all_features[indices] = F.normalize(output, dim=1).detach().cpu().numpy()
        # save labels
        all_labels[indices] = targets.numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return torch.from_numpy(all_features), torch.from_numpy(all_labels).long()


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, kk, args):
    train_features = train_features.t()
    num_test_images, imgs_per_chunk = test_labels.shape[0], args.val_batch_size
    retrieval_one_hot = torch.zeros(kk, args.num_classes)
    if torch.cuda.is_available():
        retrieval_one_hot = retrieval_one_hot.cuda(non_blocking=True)

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(range(0, num_test_images, imgs_per_chunk)),
        [batch_time, top1, top5],
        prefix=f"Test ({kk}-NN): ")

    end = time.time()
    for chunk_idx, idx in enumerate(range(0, num_test_images, imgs_per_chunk)):
        # get the features for test images
        features = test_features[idx: min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        distances = torch.mm(features, train_features)
        # distances = torch.cdist(features, train_features, p=2)
        distances, indices = distances.topk(kk, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * kk, args.num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(args.tau).exp_()
        # distances_transform = 1. / distances.clone()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, args.num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        acc1 = correct.narrow(1, 0, 1).sum().item() * 100. / targets.size(0)
        acc5 = correct.narrow(1, 0, 5).sum().item() * 100. / targets.size(0)
        top1.update(acc1, targets.size(0))
        top5.update(acc5, targets.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if chunk_idx % args.print_freq == 0:
            progress.display(chunk_idx)

    return top1.avg, top5.avg


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
