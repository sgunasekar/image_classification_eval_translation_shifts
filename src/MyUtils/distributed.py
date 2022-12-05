import datetime
import math
import os

import torch.cuda
import torch.distributed


def init_distributed_mode(args):
    """
    Use this is with torch.distributed.launch for single node training or see deit code for how to use slurm.

    dist.init_process: argument init_method='env' by default uses os.environ variable to set 'MASTER_ADDR', 'MASTER_PORT'
    """
    ngpus = torch.cuda.device_count()
    print("Number of gpus: ", ngpus)
    if ngpus<=1:
        print("No. of gpus<=1. Not using distributed mode")
        args.distributed = False
        return

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        if 'MASTER_ADDR' not in os.environ or 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
    else:
        print('Not using distributed mode. Use torchh.distributed.launch Or set RANK, WORLD_SIZE, LOCAL_RANK is os.environ')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)

    print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))

    torch.distributed.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=18000))


    torch.distributed.barrier() # blocks the processes until all processes enters this function
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def cleanup_distributed_mode():
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation. It ensures that each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch