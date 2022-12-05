from audioop import mul
from torch.autograd.grad_mode import no_grad
from MyUtils import *
from models import *
from MyUtils.config import *
import argparse

model_filename = os.path.join('..','..','..','save','checkpoint.pt')

parser = argparse.ArgumentParser(description='Translation generalization')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset to use')
parser.add_argument('--model-filename', type=str, default=model_filename, help='model filename')
parser.add_argument('--model',default='resnet18', type=str, help='model_name')
parser.add_argument('--mode', nargs='?', const='', default='', type=str, help='mode')

parser.add_argument('--debug', default=False, action='store_true',  help='debug mode (default: False)')
parser.add_argument('--batchnorm', default=False, action='store_true',  help='set model_cfg["batchnorm"]=True')
parser.add_argument('--groupnorm', default=False, action='store_true',  help='set model_cfg["groupnorm"]=True')
parser.add_argument('--batchsize', default=1024, type=int, help='evaluation batch size')


def change_test_transform_pad(testset,i,j,tr_max,mean,std):
    testset.transform = transforms.Compose([
        transforms.Pad([i,j,tr_max-1-i,tr_max-1-j], fill=tuple([min(255, int(round(255 * x))) for x in mean])),
        transforms.ToTensor(), transforms.Normalize(mean,std)])

def main(args):
    model_filename = args.model_filename
    data_dir = os.environ.get('DATA_DIR', default_cfg[args.dataset]['root'])
    save_dir = os.environ.get('OUTPUT_DIR', os.path.join(default_cfg['training']['save_dir'],'generalization'))

    if not os.path.exists(save_dir):
        print("Save directory %s does not exist: creating the directory" %(save_dir))
        os.makedirs(save_dir)

    print("save_dir", save_dir)
    print("data_dir", data_dir)

    if args.mode=='_':
        args.mode=''
    
    criterion = nn.CrossEntropyLoss().cuda()

    model_cfg, args.model = get_model_cfg(args.model)
    if model_cfg is None:
        print("Model name %s is not supported")
        raise AssertionError
    Net = model_cfg.pop("net_class")

    model_cfg['resize'] = True

    if 'BN' in args.mode or args.batchnorm:
        model_cfg['batchnorm'] = True
        args.batchnorm = True
        args.groupnorm = False
    if 'GN' in args.mode or args.groupnorm:
        model_cfg['groupnorm'] = True
        args.groupnorm = True
        args.batchnorm = False
    print(args)
        
    dataset = args.dataset.upper()    
    if args.debug:
        tr_max = 1
    elif dataset.startswith('CIFAR'):
        tr_max = 17
    elif dataset.startswith('TINYIMAGENET'):
        tr_max = 33
    else:
        raise Exception('Unsupported dataset')

    data_cfg = default_cfg[dataset]
    im_dim = data_cfg['im_dim']
    mean = data_cfg['mean']
    std = data_cfg['std']

    data_cfg['root'] = data_dir
    data_cfg['test_transform'] = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    data_cfg['transform'] = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])

    pad_size = int(0.25*im_dim)
    im_dim = im_dim+2*pad_size
    data_cfg['im_dim'] = im_dim

    model_cfg['im_dim'] = im_dim
    model_cfg['num_classes'] = data_cfg['num_classes']
    model_cfg['in_channels'] = data_cfg['in_channels']

    criterion = nn.CrossEntropyLoss().cuda()
    testlosses = np.zeros((tr_max,tr_max))
    testprec1s = np.zeros((tr_max,tr_max))
    
    print("Model: ", model_cfg)
    with torch.no_grad():
        model = Net(model_cfg)
        model = model.cuda()
        print(model) 
        model.eval()
        
    trained_model = torch.load(model_filename, map_location='cpu')
    model.load_state_dict(trained_model['model_state_dict'])
    
    print("Dataset:", data_cfg)
    data = Dataset(data_cfg=data_cfg, download=False, val_only=True, print_cfg=False)
    batch_size = args.batch_size
    testloader = torch.utils.data.DataLoader(data.testset,shuffle=False, batch_size=batch_size, pin_memory=True,  drop_last=False, num_workers=32)
    
    for i in range(tr_max):
        for j in range(tr_max):
            
            change_test_transform_pad(testloader.dataset,i,j, tr_max, data_cfg["mean"], data_cfg["std"])

            epoch_test_stats = validate(testloader, model, criterion=criterion, device='cuda')
            testprec1s[tr_max-j-1,i] = epoch_test_stats['prec1']
            testlosses[tr_max-j-1,i] = epoch_test_stats['loss']
        print(i)

    np.save(
        os.path.join(save_dir,"translation_%s_%s_%s.npy" %(args.dataset,args.model,args.mode)),
        {
            'testlosses': testlosses,
            'testprec1s': testprec1s
        }
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)