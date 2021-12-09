import argparse
import os
import random
import time
import warnings
import utils
import sys
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as adjusted_nmi
from sklearn.metrics import adjusted_rand_score as adjusted_rand_index
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Kmeans Evaluation')
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
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--label-subset', default="10", type=str, choices=["1", "10", "100"],
                    help='percentage of labeled data: 1%, 10% or 100% (default: 1)')
parser.add_argument('--num-classes', default=1000, type=int,
                    help='number of classes (1000 for ImageNet, 10 (default: 0.1)')
parser.add_argument('--load-features', action='store_true',
                    help='use features from earlier dump (in args.save_path)')
parser.add_argument('--kk', default=1000, type=int,
                    help='number of clusters to use for kmeans (default: 1000)')
parser.add_argument('--model', default='mocov2', const='mocov2', nargs='?',
                    choices=['mocov2', 'swav', 'simsiam', 'barlowtwins', 'obow', 'dino'],
                    help='type of pretrained model (default: %(default)s)')
parser.add_argument('--backbone-dim', default=2048, type=int,
                    help='backbone dimension size (default: %(default)s)')


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
    if args.model == 'barlowtwins':
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    elif args.model == 'dino':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    else:
        model = models.__dict__[args.arch]()
    model.fc = nn.Identity()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # load state dictionary
            if args.model == 'swav':
                state_dict = checkpoint
            elif args.model == 'obow':
                state_dict = checkpoint['network']
            else:
                state_dict = checkpoint['state_dict']

            # fix prefix
            for k in list(state_dict.keys()):

                if args.model == 'mocov2':
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        new_k = k[len('module.encoder_q.'):]
                        state_dict[new_k] = state_dict[k]
                        del state_dict[k]

                elif args.model == 'simsiam':
                    if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                        new_k = k[len('module.encoder.'):]
                        state_dict[new_k] = state_dict[k]
                        del state_dict[k]

                elif args.model == 'swav':
                    if k.startswith('module') and not k.startswith('module.projection_head'):
                        new_k = k[len('module.'):]
                        state_dict[new_k] = state_dict[k]
                        del state_dict[k]

                elif args.model == 'obow':
                    if k.startswith('fc'):
                        del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert len(msg.missing_keys) == 0,  "missing_keys: {}".format(msg.missing_keys)
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

    train_dataset = utils.ImageFolderWithIndices(traindir, transform)
    if args.label_subset != "100":
        train_dataset = utils.imagenet_subset_samples(train_dataset, traindir, args.label_subset)  # extract subset
    val_dataset = utils.ImageFolderWithIndices(valdir, transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.load_features:
        train_features = np.load(os.path.join(args.save_path, "trainfeat.npy"))
        val_features = np.load(os.path.join(args.save_path, "valfeat.npy"))
        val_labels = np.load(os.path.join(args.save_path, "vallabels.npy"))
    else:
        train_features, _ = inference(train_loader, model, args, prefix='Train Set Inference: ')
        val_features, val_labels = inference(val_loader, model, args, prefix='Test Set Inference: ')

        # dump
        np.save(os.path.join(args.save_path, "trainfeat"), train_features)
        np.save(os.path.join(args.save_path, "valfeat"), val_features)
        np.save(os.path.join(args.save_path, "vallabels"), val_labels)

    # evaluate kmeans classifier
    print("Features are ready!\nEvaluate K-Means Classifier.")
    kmeans_classifier(train_features, val_features, val_labels, args)


@torch.no_grad()
def inference(loader, model, args, prefix):
    all_features = np.zeros((len(loader.dataset), args.backbone_dim), dtype=np.float)
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
        all_features[indices] = output.detach().cpu().numpy()
        # save labels
        all_labels[indices] = targets.numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return all_features, all_labels


@torch.no_grad()
def kmeans_classifier(train_features, val_features, targets, args):
    # fit based on train set
    print('=> fitting K-Means classifier..')
    kmeans = KMeans(n_clusters=args.kk, verbose=True, n_jobs=-1).fit(train_features)

    # save kmeans model
    print('=> saving K-Means classifier..')
    kmeans_save_path = os.path.join(args.save_path, args.model + '_kmeans.pkl')
    pickle.dump(kmeans, open(kmeans_save_path, "wb"))

    # predict
    preds = kmeans.predict(val_features)

    # evaluate
    val_nmi = nmi(targets, preds)
    val_adjusted_nmi = adjusted_nmi(targets, preds)
    val_adjusted_rand_index = adjusted_rand_index(targets, preds)
    print('=> number of samples: {}'.format(len(targets)))
    print('=> number of unique assignments: {}'.format(len(set(preds))))
    print('=> NMI: {:.3f}%'.format(val_nmi * 100.0))
    print('=> Adjusted NMI: {:.3f}%'.format(val_adjusted_nmi * 100.0))
    print('=> Adjusted Rand-Index: {:.3f}%'.format(val_adjusted_rand_index * 100.0))

    # compute accuracy
    num_classes = max(targets.max(), preds.max()) + 1
    count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for ii in range(preds.shape[0]):
        count_matrix[preds[ii], targets[ii]] += 1
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]

    if preds.max() > targets.max():  # if using over-clustering, append remaining clusters to best option
        for cls_idx in range(targets.max(), preds.max()):
            reassignment[cls_idx, 1] = count_matrix[cls_idx].argmax()

    acc = count_matrix[reassignment[:, 0], reassignment[:, 1]].sum().astype(np.float32) / preds.shape[0]
    print('=> Accuracy: {:.3f}%'.format(acc * 100.0))


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
