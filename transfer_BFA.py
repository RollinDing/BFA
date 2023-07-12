from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, clustering_loss, change_quan_bitwidth
from tensorboardX import SummaryWriter
import models as models
from models.quantization import quan_Conv2d, quan_Linear, quantize

from attack.BFA import *
from attack.dual_model_BFA import *
import torch.nn.functional as F
import copy

import pandas as pd
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='/home/elliot/data/pytorch/svhn/',
                    type=str,
                    help='Path to dataset')

parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'usps'],
    help='Choose between Cifar10/100 and ImageNet.')

parser.add_argument(
    '--std-dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'usps'],
    help='Choose between Cifar10/100 and ImageNet as student downstream dataset.')

parser.add_argument('--arch',
                    metavar='ARCH',
                    default='lbcnn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')

# Optimization options
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')

parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])

parser.add_argument('--percent', 
                    type=float,
                    default=0.5,
                    help='percentage of training data to use')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')

parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')

parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')

parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)

# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')

parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')

parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--resume_student',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)

parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')

parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')

parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')

# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')

# quantization
parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')

parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')

# Bit Flip Attack
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')

parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')

parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')

parser.add_argument(
    '--k_top',
    type=int,
    default=None,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)

parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')

# Piecewise clustering
parser.add_argument('--clustering',
                    dest='clustering',
                    action='store_true',
                    help='add the piecewise clustering term.')

parser.add_argument('--lambda_coeff',
                    type=float,
                    default=1e-3,
                    help='lambda coefficient to control the clustering term')

##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

###############################################################################
###############################################################################


def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init dataset
    train_loader_teacher, test_loader_teacher, num_classes = init_teacher_dataset(args)

    # Init teacher model, criterion, and optimizer
    net_teacher = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net_teacher), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net_teacher = torch.nn.DataParallel(net_teacher, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net_teacher.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net_teacher.named_parameters() if 'step_size' in name 
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'],
                                    weight_decay=state['decay'],
                                    nesterov=True)
    if args.use_cuda:
        net_teacher.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net_teacher.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net_teacher.load_state_dict(state_tmp)

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                      log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)
        
    # Configure the quantization bit-width
    if args.quan_bitwidth is not None:
        change_quan_bitwidth(net_teacher, args.quan_bitwidth)

    # update the step_size once the model is loaded. This is used for quantization.
    for m in net_teacher.modules():
        if isinstance(m, torch.nn.BatchNorm2d):  # Skip BatchNorm2d layers
            continue
        if hasattr(m, 'weight'):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for weight reset
    if args.reset_weight:
        for m in net_teacher.modules():
            if isinstance(m, torch.nn.BatchNorm2d):  # Skip BatchNorm2d layers
                continue
            if hasattr(m, 'weight'):
                m.__reset_weight__()

    # initialize student dataset 
    train_loader_student, test_loader_student, num_classes_student = init_student_dataset(args)

    # Init student model, criterion, and optimizer
    net_student = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net_student), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net_student = torch.nn.DataParallel(net_student, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net_student.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net_student.named_parameters() if 'step_size' in name 
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'],
                                    weight_decay=state['decay'],
                                    nesterov=True)
    if args.use_cuda:
        net_student.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    if args.resume_student:
        if os.path.isfile(args.resume_student):
            print_log("=> loading checkpoint '{}'".format(args.resume_student), log)
            checkpoint = torch.load(args.resume_student)
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net_student.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net_student.load_state_dict(state_tmp)

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume_student, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume_student),
                      log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)
        
    # Configure the quantization bit-width
    if args.quan_bitwidth is not None:
        change_quan_bitwidth(net_student, args.quan_bitwidth)

    for m in net_student.modules():
        if isinstance(m, torch.nn.BatchNorm2d):  # Skip BatchNorm2d layers
            continue
        if hasattr(m, 'weight'):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for weight reset
    if args.reset_weight:
        for m in net_student.modules():
            if isinstance(m, torch.nn.BatchNorm2d):  # Skip BatchNorm2d layers
                continue
            if hasattr(m, 'weight'):
                m.__reset_weight__()

    # Evaluate teacher model
    # _,_,_, output_summary = validate(test_loader, net, criterion, log, summary_output=True)
    # pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary_{}.csv'.format(args.arch)),
    #                                     header=['top-1 output'], index=False)

    # # Evaluate student model
    # _,_,_, output_summary = validate(test_loader_student, net_student, criterion, log, summary_output=True)
    # pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'student_output_summary_{}.csv'.format(args.arch)),
    #                                     header=['top-1 output'], index=False)
    
    # Pretrained feature extractor is the layer before the classifier module


    # for m in feature_extractor.children():
    #     if isinstance(m, torch.nn.BatchNorm2d):  # Skip BatchNorm2d layers
    #         continue
    #     if hasattr(m, 'weight'):
    #         print(m.weight)

    # # There is batch normalization in the feature extractor, so the performance is various for different dataset
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # for i, (input, target) in enumerate(test_loader_student):
    #     if args.use_cuda:
    #         target = target.cuda()
    #         input = input.cuda()

    #     # compute output
    #     feature = feature_extractor(input)
    #     feature = feature.view(feature.size(0), -1)
    #     output  = classifierA(feature)
        
    #     # measure accuracy and record loss
    #     prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    #     top1.update(prec1.item(), input.size(0))
    #     top5.update(prec5.item(), input.size(0))
    # print_log(
    #     '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
    #     .format(top1=top1, top5=top5, error1=100 - top1.avg), log)    
    

    attacker = DualModelBFA(criterion, net_student, args.k_top)
    # attacker = BFA(criterion, net_student, args.k_top)
    net_clean_student = copy.deepcopy(net_student)
    net_clean_teacher = copy.deepcopy(net_teacher)

    # weight_conversion(net)
    feature_extractor = net_clean_student.features
    classifier_teacher = net_clean_teacher.classifier
    classifier_student = net_clean_student.classifier

    if args.enable_bfa:
        perform_attack(attacker, feature_extractor, classifier_student, net_clean_student, train_loader_student, test_loader_student,
                    classifier_teacher, net_clean_teacher, train_loader_teacher, test_loader_teacher,
                    args.n_iter, log, writer, csv_save_path=args.save_path,
                    random_attack=args.random_bfa)
        return
    log.close()

def init_teacher_dataset(args):
    # Init teacher dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist' or args.dataset == 'usps':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    elif args.dataset == 'mnist' or args.dataset == 'usps':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to match the CIFAR10 dataset
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to match the CIFAR10 dataset
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'usps':
        train_data = dset.USPS(args.data_path,
                            train=True,
                            transform=train_transform,
                            download=True)
        test_data = dset.USPS(args.data_path,
                            train=False,
                            transform=test_transform,
                            download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    return train_loader, test_loader, num_classes

def init_student_dataset(args):
    # Init student dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.std_dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.std_dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.std_dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.std_dataset == 'mnist' or args.std_dataset == 'usps':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.std_dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.std_dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    elif args.std_dataset == 'mnist' or args.std_dataset == 'usps':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to match the CIFAR10 dataset
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to match the CIFAR10 dataset
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.std_dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.std_dataset == 'usps':
        train_data = dset.USPS(args.data_path,
                               train=True,
                               transform=train_transform,
                               download=True)
        test_data = dset.USPS(args.data_path,
                              train=False,
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.std_dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.std_dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.std_dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.std_dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.std_dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    percent = args.percent
    print(f"=> Preparing data for student model")

    # Randomly select a subset of the training data
    if percent < 1:
        indices = torch.randperm(len(train_data))[:int(len(train_data) * percent)]
        train_data = torch.utils.data.Subset(train_data, indices)
        print(f"=> Using {percent * 100}% of the training data")
    else:
        print(f"=> Using all training data")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    return train_loader, test_loader, num_classes

def perform_attack(attacker, feature_extractor, classifier_student, model_student_clean, train_loader_student, test_loader_student,
                   classifier_teacher, model_teacher_clean, train_loader_teacher, test_loader_teacher,
                   N_iter, log, writer, csv_save_path=None, random_attack=False):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    feature_extractor.eval()
    feature_extractor_clean = copy.deepcopy(feature_extractor)
    classifier_student.eval()
    model_student_clean.eval()
    classifier_teacher.eval()
    model_teacher_clean.eval()

    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()

    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader_student):
        if args.use_cuda:
            target = target.cuda()
            data = data.cuda()
        # Override the target to prevent label leaking
        data_student = data
        feature_student = feature_extractor(data_student)
        feature_student = feature_student.view(feature_student.size(0), -1)
        output_student = classifier_student(feature_student)
        _, target_student = output_student.data.max(1)
        break

    for _, (data, target) in enumerate(train_loader_teacher):
        if args.use_cuda:
            target = target.cuda()
            data = data.cuda()
        # Override the target to prevent label leaking
        data_teacher = data
        feature_teacher = feature_extractor(data_teacher)
        feature_teacher = feature_teacher.view(feature_teacher.size(0), -1)
        output_teacher = classifier_teacher(feature_teacher)
        _, target_teacher = output_teacher.data.max(1)
        break

    # evaluate the test accuracy of clean student model
    val_acc_top1_student, val_acc_top5_student, val_loss_student, output_summary = validate_feature_extractor(test_loader_student, feature_extractor, classifier_student,
                                                    attacker.criterion, log, summary_output=True)
    val_acc_top1_teacher, val_acc_top5_teacher, val_loss_teacher, output_summary = validate_feature_extractor(test_loader_teacher, feature_extractor, classifier_teacher,
                                                    attacker.criterion, log, summary_output=True)
    
    tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
    tmp_df['BFA iteration'] = 0
    tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_0.csv'.format(args.arch)),
                                        index=False)

    writer.add_scalar('attack/val_top1_acc_student', val_acc_top1_student, 0)
    writer.add_scalar('attack/val_top5_acc_student', val_acc_top5_student, 0)
    writer.add_scalar('attack/val_loss_student', val_loss_student, 0)

    writer.add_scalar('attack/val_top1_acc_teacher', val_acc_top1_teacher, 0)
    writer.add_scalar('attack/val_top5_acc_teacher', val_acc_top5_teacher, 0)
    writer.add_scalar('attack/val_loss_teacher', val_loss_teacher, 0)


    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()
    
    df = pd.DataFrame() #init a empty dataframe for logging
    last_val_acc_top1_student = val_acc_top1_student
    
    for i_iter in range(N_iter):
        print_log('**********************************', log)
        attack_log = attacker.dual_model_progressive_bit_search(feature_extractor, classifier_student, classifier_teacher, 
                                                                data_student, data_teacher, 
                                                                target_student, target_teacher)
            
        
        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        h_dist = hamming_distance(feature_extractor, feature_extractor_clean)

        # record the loss
        if hasattr(attacker, "loss_max"):
            losses.update(attacker.loss_max, data.size(0))

        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)
        try:
            print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                    log)
            print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
        except:
            pass
        
        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        print_log('hamming_dist: {:.0f}'.format(h_dist), log)

        writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
        writer.add_scalar('attack/h_dist', h_dist, i_iter + 1)
        writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

        # exam the BFA on entire val dataset
        val_acc_top1_student, val_acc_top5_student, val_loss_student, output_summary = validate_feature_extractor(test_loader_student, feature_extractor, classifier_student,
                                                        attacker.criterion, log, summary_output=True)
        val_acc_top1_teacher, val_acc_top5_teacher, val_loss_teacher, output_summary = validate_feature_extractor(test_loader_teacher, feature_extractor, classifier_teacher,
                                                        attacker.criterion, log, summary_output=True)
        tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
        tmp_df['BFA iteration'] = i_iter + 1
        tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_{}.csv'.format(args.arch, i_iter + 1)),
                                    index=False)
    
        
        # add additional info for logging
        acc_drop = last_val_acc_top1_student - val_acc_top1_student
        last_val_acc_top1_student = val_acc_top1_student
        
        # print(attack_log)
        for i in range(attack_log.__len__()):
            attack_log[i].append(val_acc_top1_student)
            attack_log[i].append(acc_drop)
        # print(attack_log)
        df = df.append(attack_log, ignore_index=True)

        writer.add_scalar('attack/val_top1_acc_student', val_acc_top1_student, i_iter + 1)
        writer.add_scalar('attack/val_top5_acc_student', val_acc_top5_student, i_iter + 1)
        writer.add_scalar('attack/val_loss_student', val_loss_student, i_iter + 1)

        writer.add_scalar('attack/val_top1_acc_teacher', val_acc_top1_teacher, i_iter + 1)
        writer.add_scalar('attack/val_top5_acc_teacher', val_acc_top5_teacher, i_iter + 1)
        writer.add_scalar('attack/val_loss_teacher', val_loss_teacher, i_iter + 1)

        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()
        
        # Stop the attack if the accuracy is below the configured break_acc.
        if args.dataset == 'cifar10':
            break_acc = 11.0
        elif args.dataset == 'svhn':
            break_acc = 11.0
        elif args.dataset == 'mnist' or 'usps':
            break_acc = 11.0
        elif args.dataset == 'cifar100':
            break_acc = 1.1
        elif args.dataset == 'imagenet':
            break_acc = 0.2
        if val_acc_top1_student <= break_acc:
            break
        
    # attack profile
    column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                  'weight before attack', 'weight after attack', 'validation accuracy',
                  'accuracy drop']
    df.columns = column_list
    df['trial seed'] = args.manualSeed
    if csv_save_path is not None:
        csv_file_name = 'attack_profile_{}.csv'.format(args.manualSeed)
        export_csv = df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)

    return

def validate_feature_extractor(val_loader, feature_extractor, classifier, criterion, log, summary_output=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    feature_extractor.eval()
    classifier.eval()
    output_summary = [] # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            feature = feature_extractor(input)
            feature = feature.view(feature.size(0), -1)
            output = classifier(feature)
            loss = criterion(output, target)
            
            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy() # get the index of the max log-probability
                output_summary.append(tmp_list)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
        
    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary
    else:
        return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, log, summary_output=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            
            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy() # get the index of the max log-probability
                output_summary.append(tmp_list)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
        
    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary
    else:
        return top1.avg, top5.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
