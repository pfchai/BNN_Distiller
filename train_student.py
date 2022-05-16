# -*- coding: utf-8 -*-

import os
import argparse
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from dataset.dataset import load_data
from models import model_dict
from models.bin_resnetv2 import BinarizeConv2d
from distiller.VID import VIDLoss
from distiller.KD import DistillKL
from utils.loops import train_layer as train, validate
from utils.utils import adjust_learning_rate, show_model_structure, freeze_model


def cpt_tau(epoch, epochs, tau_min, tau_max):
    "compute tau"
    a = torch.tensor(np.e)
    T_min, T_max = torch.tensor(tau_min).float(), torch.tensor(tau_max).float()
    A = (T_max - T_min) / (a - 1)
    B = T_min - A
    tau = A * torch.tensor([torch.pow(a, epoch / epochs)]).float() + B
    return tau


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--gpus', type=str, default='', help='gpu')
    parser.add_argument('--start_step', default=1, type=int, help='start step, 1-5 only train part of model, 6: full train', choices=[1, 2, 3, 4, 5, 6])

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet18_1w1a', choices=['resnet18_1w1a'])
    parser.add_argument('--path_t', type=str, default='./save/models/resnet18_cifar10_lr_0.05_decay_0.0005_trial_0/resnet18_last.pth', help='teacher model snapshot')

    parser.add_argument('--trial', type=str, default='1', help='trial id')

    # distillation
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='weight balance for KD')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    # Binarize
    parser.add_argument('--tau_min', default=0.85, type=float, help='tau_min')
    parser.add_argument('--tau_max', default=0.99, type=float, help='tau_max')

    opt = parser.parse_args()
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_r:{}_a:{}_{}'.format(
        opt.model_s, opt.model_t, opt.dataset,
        opt.gamma, opt.alpha, opt.trial
    )

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    n_cls = 10
    if opt.dataset == 'cifar100':
        n_cls = 100
    elif opt.dataset == 'cifar10':
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    train_loader, val_loader, n_data = load_data(
        dataset=opt.dataset,
        data_path=opt.data_path,
        batch_size=opt.batch_size,
        batch_size_test=opt.batch_size,
        num_workers=opt.num_workers,
        is_instance=True
    )

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    print('model teacher')
    show_model_structure(model_t)
    print('model student')
    show_model_structure(model_s)

    criterion_cls = nn.CrossEntropyLoss()

    # every step vid criterion
    criterion_vids = []
    for step in range(1, 6):
        t_out = model_t(data, step)
        s_out = model_s(data, step)
        criterion_vids.append(VIDLoss(s_out.shape[1], t_out.shape[1], t_out.shape[1]))

    layer_names = {
        'conv1': 1,
        'bn1': 1,
        'layer1': 2,
        'layer2': 3,
        'layer3': 4,
        'layer4': 5,
        'linear': 6,
        'bn2': 6,
    }

    # validate teacher accuracy
    if torch.cuda.is_available():
        model_t.cuda()
        criterion_cls.cuda()
        cudnn.benchmark = True

    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    for step in range(opt.start_step, 7):
        trainable_list = nn.ModuleList([model_s])
        to_freeze = []
        if step < 6:
            criterion = criterion_vids[step - 1]
            trainable_list.append(criterion)
            # to freeze
            to_freeze = [k for k, v in layer_names.items() if v < step]

        else:
            criterion = DistillKL(opt.kd_T)

        model_s = freeze_model(model_s, to_freeze)

        # optimizer
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, trainable_list.parameters()),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay
        )

        criterion_list = nn.ModuleList([criterion_cls, criterion])

        # append teacher after optimizer to avoid weight_decay
        if torch.cuda.is_available():
            model_s.cuda()
            model_t.cuda()
            criterion_list.cuda()
            cudnn.benchmark = True

        # record names of conv_modules
        conv_modules = []
        for name, module in model_s.named_modules():
            if isinstance(module, BinarizeConv2d):
                conv_modules.append(module)

        epochs = 0
        if step < 6:
            epochs = opt.init_epochs
        else:
            epochs = opt.epochs

        # routine
        for epoch in range(1, epochs + 1):

            adjust_learning_rate(epoch, opt, optimizer)
            # print("==> training step {}...".format(step))

            # compute threshold tau
            tau = cpt_tau(epoch, epochs, opt.tau_min, opt.tau_max)
            for module in conv_modules:
                if torch.cuda.is_available():
                    module.tau = tau.cuda()
                else:
                    module.tau = tau

            time1 = time.time()
            train_acc, train_loss = train(epoch, train_loader, model_t, model_s, criterion_list, optimizer, opt, step)
            time2 = time.time()
            print('step {}, epoch {}, total time {:.2f}'.format(step, epoch, time2 - time1))

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

            test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            logger.log_value('test_acc_top5', tect_acc_top5, epoch)

            # save the best model
            if step == 7:
                if test_acc > best_acc:
                    best_acc = test_acc
                    state = {
                        'epoch': epoch,
                        'model': model_s.state_dict(),
                        'best_acc': best_acc,
                    }
                    save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                    print('saving the best model!')
                    torch.save(state, save_file)

                # regular saving
                if epoch % opt.save_freq == 0:
                    print('==> Saving...')
                    state = {
                        'epoch': epoch,
                        'model': model_s.state_dict(),
                        'accuracy': test_acc,
                    }
                    save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                    torch.save(state, save_file)

        # This best accuracy is only for printing purpose.
        # The results reported in the paper/README is from the last epoch.
        print('best accuracy:', best_acc)

        # save model
        state = {
            'opt': opt,
            'model': model_s.state_dict(),
        }
        save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
        torch.save(state, save_file)


if __name__ == '__main__':
    main()
