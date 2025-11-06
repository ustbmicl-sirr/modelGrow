from timm.optim import create_optimizer_v2
import torch.optim as optim
import numpy as np
import math


def get_optimizer(net, args):
    if 'sgd' == args.optimizer:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif 'adam' == args.optimizer:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif 'adamw' == args.optimizer:
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif 'amsgrad' == args.optimizer:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif 'adagrad' == args.optimizer:
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif 'adadelta' == args.optimizer:
        optimizer = optim.Adadelta(net.parameters(), weight_decay=args.weight_decay)
    elif 'rmsprop' == args.optimizer:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, weight_decay=args.weight_decay)
    elif 'lion' == args.optimizer:
        optimizer = create_optimizer_v2(net.parameters(), opt="lion", lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Unknown --optimizer')
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def decay_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1


def cosine_scheduler(base_lr, final_lr, epochs, niter_per_ep):
    iters = np.arange(epochs * niter_per_ep)
    schedule = np.array([
        final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters
    ])
    assert len(schedule) == epochs * niter_per_ep
    return schedule
