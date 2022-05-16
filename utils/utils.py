# -*- coding: utf-8 -*-

import torch
import numpy as np


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def show_model_structure(model):
    print("model structure: ")
    for name, module in model._modules.items():
        print('\t' + str(name) + ': ' + str(module))
    num_parameters = sum([layer.nelement() for layer in model.parameters()])
    print("number of parameters: %d", num_parameters)


def freeze_model(model, to_freeze_dict):
    for (name, param) in model.named_parameters():
        for fn in to_freeze_dict:
            if name.startswith(fn):
                param.requires_grad = False
    return model


def show_model_params(model, step, epoch):
    with open('params_{}_{}.data'.format(step, epoch), 'w') as f:
        for (name, param) in model.named_parameters():
            f.write(name + ': ' + str(param.requires_grad))
            d = [i.item() for i in param.reshape(-1)[:5]]
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(*d))
            # print(name, ': ', [i.item() for i in param.reshape(-1)[:3]])
