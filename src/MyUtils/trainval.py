import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.cuda.amp import GradScaler

from MyUtils.mixup import Mixup

from .utils import *


def get_train_kwargs(train_kwargs, args, suffix):
    num_classes = train_kwargs.pop('num_classes')
    mixup_active = args.use_mixup and (args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None)
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)
        train_kwargs['mixup_fn'] = mixup_fn
    if args.clip_grad:
        train_kwargs['clip_grad'] = args.clip_grad

    if "resnet" in args.model:
        if args.groupnorm:
            suffix = suffix+"_GN"
        elif args.batchnorm:
            suffix = suffix+"_BN"

    train_kwargs['print_freq'] = args.print_freq
    if args.use_amp:
        loss_scaler = GradScaler()
        train_kwargs['loss_scaler'] = loss_scaler
    else:
        train_kwargs['loss_scaler'] = None

    return train_kwargs, suffix

## Training
# Separating out val iter & train iter for memory reasons -- there is 2x overheard otherwise. For some reason without a separate function the memory assignment images,labels is not recaptured effectively
## e.g., resnet18_b512 with val_iter, all iters have
# before iter 44781056
# inside iter (after moving new images,labels to cuda) 59486720
# after iter 44781056
## resnet18_b512 without val_iter
# before iter 44781056
# inside iter 59465216
# after iter 59485696
# before iter 59485696
# inside iter 73645568
# OOM error
## resnet18_b256 without val_iter -> does not help even with del images, labels, output
# before iter 44781056
# inside iter 52778496
# after iter 52788736
# before iter 52788736
# inside iter 59868672
# after iter 59868672
# before iter 52788736
# inside iter 59868672
# after iter 59868672
def train_epoch(trainloader, model, optimizer, epoch, criterion, device, print_freq=100,mixup_fn=None, loss_scaler=None, clip_grad=None, SW=None):
    
    
    def train_iter(images,labels, print_mem=False, is_first_batch = False):

        gradient_clipped = False
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if print_mem: print("\t\t Memory: in iter after loading data %d " %torch.cuda.memory_allocated(), ' imsize: ', images.shape, ' labelsize: ', labels.shape)
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        if print_mem: print("\t\t Memory: in iter after mixup", torch.cuda.memory_allocated())

        optimizer.zero_grad()
        
        if not(epoch) and is_first_batch and SW is not None:
            import torchvision            
            samples = images[:8]
            SW.add_graph(model.module,samples)   
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            unnormalize = torchvision.transforms.Normalize(-mean/std,1.0/std)
            samples = unnormalize(samples)
            grid = torchvision.utils.make_grid(samples)
            SW.add_image('train images', grid, 0)
            print("Added graph and sample images to tensorboard")

        with torch.cuda.amp.autocast(enabled=(loss_scaler is not None)):
            output = model(images)
            if print_mem: print("\t\t Memory: in iter after model output", torch.cuda.memory_allocated())
            loss = criterion(output, labels)
            if np.isnan(loss.item()):
                kwargs = {'force':True}
                print("\t\t Loss is NaN. Breaking in gpu", next(model.parameters()).device, **kwargs)
                print(output)
                for i in range(len(output)):
                    print(output[i], **kwargs)

                raise ValueError("Loss is NaN")

        if not loss_scaler:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=False)
            optimizer.step()
        else:
            loss_scaler.scale(loss).backward()
            if clip_grad is not None:
                loss_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=False)
                if grad_norm >= clip_grad: 
                    gradient_clipped = True
            loss_scaler.step(optimizer)
            loss_scaler.update()

        loss = loss.item()
        with torch.no_grad():
            if mixup_fn is None: # compute precision only if mixup is off
                output_data = output.data
                prec1 = accuracy(output_data, labels)[0]
                prec1 = prec1.item()
            else:
                prec1 = None

        if print_mem: print("\t\t Memory: end of iter", torch.cuda.memory_allocated())

        return loss, prec1, gradient_clipped


    # setup
    trainloss = AverageMeter()
    traintop1 = AverageMeter()
    grad_clip_count = AverageMeter()

    if loss_scaler is not None and ((not epoch) and distributed.is_main_process()): print("Using AMP")

    model.train()

    if epoch<5: print("Memory: epoch start", torch.cuda.memory_allocated())

    # iterations
    for i, (images, labels) in enumerate(trainloader):

        print_mem = i<5 and epoch<5

        loss, prec1, gradient_clipped = train_iter(images, labels, print_mem, is_first_batch = not(i))

        if gradient_clipped: grad_clip_count.update(1,1)
        if prec1 is not None: traintop1.update(prec1, images.size(0))
        trainloss.update(loss, images.size(0))

        if (i+1) % print_freq == 0:
            print_progress('[%d][%d/%d]' %(epoch, i+1, len(trainloader)), loss=trainloss, top1=traintop1)

    # synchronize
    trainloss.synchronize_between_processes()
    if mixup_fn is None: traintop1.synchronize_between_processes()
    if clip_grad is not None: grad_clip_count.synchronize_between_processes()

    lr = get_lr(optimizer)
    grad_clip_count = grad_clip_count.count
    param_norm, grad_norm = get_param_grad_norm(model)
    cuda_alloc_memory = torch.cuda.memory_allocated()

    print_progress('Epoch: %d (lr=%0.2g, memory=%d)' %(epoch, lr, cuda_alloc_memory), loss=trainloss, top1=traintop1)
    if mixup_fn is not None and not(epoch): print("Note: Train accuracy is not computed when mixup is on")

    epoch_train_stats = dict(loss=trainloss.avg(), prec1=traintop1.avg(), lr=lr, grad_clip_count=grad_clip_count, param_norm=param_norm, grad_norm=grad_norm, cuda_alloc_memory=cuda_alloc_memory)

    return epoch_train_stats



def validate(testloader, model,  device, criterion=nn.CrossEntropyLoss()):

    def val_iter(images,labels):
        with torch.no_grad():
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, labels)

            prec1 = accuracy(output, labels)[0]

            loss = loss.item()
            prec1 = prec1.item()

            return loss,prec1

    testloss = AverageMeter()
    testtop1 = AverageMeter()

    model.eval()

    for i, (images, labels) in enumerate(testloader):

        loss, prec1 = val_iter(images,labels)#,model,device,criterion)

        testloss.update(loss, images.size(0))
        testtop1.update(prec1, images.size(0))

    testloss.synchronize_between_processes()
    testtop1.synchronize_between_processes()

    print_progress('Validation:', loss=testloss, top1=testtop1)

    epoch_test_stats = dict(loss=testloss.avg(), prec1=testtop1.avg())

    return epoch_test_stats


def get_criterion(args):
    criterion = nn.CrossEntropyLoss()

    if args.use_mixup and args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.use_label_smoothing and args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    return criterion