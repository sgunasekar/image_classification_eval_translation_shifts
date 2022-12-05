import random

import numpy as np
import torch

##############################
from . import distributed


def set_random_seed(s=0):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)

def print_progress(prefix, loss, top1):
    print(f'{prefix}\t'
          f'Loss {loss.avg():.4f}\t'
          f'Prec@1 {top1.avg():.3f}',flush=True)
    #print(f'{prefix}\t'
    #      'Time {batchtime.sum:.3f} ({batchtime.avg():.3f})\t'
    #      'Loss {loss.avg():.4f}\t'
    #      'Prec@1 {top1.avg():.3f}'.format(prefix=prefix, batchtime=batchtime, loss=loss, top1=top1),flush=True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_param_grad_norm(model):
    with torch.no_grad():
        paramnorm = torch.norm(torch.stack([torch.norm(p) for p in model.parameters()]))
        gradnorm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))
    return paramnorm.item(), gradnorm.item()
    # gradnorm = 0
    # for param in model.parameters():
    #     paramnorm += torch.norm(param)**2
    #     gradnorm += torch.norm(param.grad.data)**2
    # return math.sqrt(paramnorm), math.sqrt(gradnorm)

def scale_params(model,scale):
    for param in model.parameters():
        param.data.mul_(scale)

def count_all_parameters(model):
    return sum(param.numel() for param in model.parameters())

def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

class AverageMeter():
    """
    Computes and stores the average and current value
    Source: https://github.com/chengyangfu/pytorch-vgg-cifar10"
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def avg(self):
        if not(self.count): return 0
        return self.sum/self.count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the val!
        """
        if not distributed.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]

    def __str__(self):
        return "No. samples: %d, sum: %0.4f, avg: %0.4f" %(self.count, self.sum, self.avg())


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    Source: https://github.com/chengyangfu/pytorch-vgg-cifar10
    """

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

def load_checkpoint(args, model_without_ddp, optimizer, scheduler, loss_scaler):

    checkpoint_file = args.resume_checkpoint
    if checkpoint_file is None:
        print("Warning: 'cfg[training][resume]' is set to True but no valid checkpoint file provided:ignoring resume flag")
        args.resume = False
    else:
        checkpoint = torch.load(checkpoint_file,map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if args.use_amp: loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])

        start_epoch = checkpoint['epoch']+1
        train_stats = checkpoint['train_stats']
        test_stats = checkpoint['test_stats']
        best_testprec1 = checkpoint['best_testprec1']
        best_testprec1_epoch = checkpoint['best_testprec1_epoch']
        scheduler.step(start_epoch)

        print("Resuming from checkpoint saved at epoch %d: train loss: %0.4f, train prec@1: %0.4f, lr: %0.4g"\
            %(start_epoch-1,train_stats['losses'][-1],train_stats['losses'][-1],train_stats['lrs'][-1]))

    return start_epoch, train_stats, test_stats, best_testprec1, best_testprec1_epoch
