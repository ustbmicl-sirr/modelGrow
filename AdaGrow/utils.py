import os
import torch
import torch.distributed as dist
import argparse


def tuple_of_ints(string):
    try:
        return tuple(map(int, string.strip('()').split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for tuple_of_ints: {string}")


def list_to_str(l):
    list(map(str, l))
    s = ''
    for v in l:
        s += str(v) + '-'
    return s[:-1]


def params_id_to_name(net):
    themap = {}
    for i, (k, v) in enumerate(net.named_parameters()):
        themap[i] = k
    return themap


def params_name_to_id(net):
    themap = {}
    for i, (k, _) in enumerate(net.named_parameters()):
        themap[k] = i
    return themap


def save_all(epoch, train_accu, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'train_accu': train_accu,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'name_id_map': params_name_to_id(model),
    }, path)
    

class MovingMaximum(object):
    def __init__(self):
        self.data = [] # data[i] is the maximum val in data[0:i+1]
        self.max = 0.0

    def push(self, current_data):
        if len(self.data) == 0:
            self.max = current_data
        elif current_data > self.max:
            self.max = current_data
        self.data.append(self.max)

    def get(self):
        return self.data

    def delta(self, start, end):
        try:
            res = self.data[end] - self.data[start]
        except IndexError:
            res = self.data[end]
        return res


class ExponentialMovingAverage(object):
    def __init__(self, decay=0.95):
        self.data = []
        self.decay = decay
        self.avg_val = 0.0

    def push(self, current_data):
        self.avg_val = self.decay * self.avg_val + (1 - self.decay) * current_data
        self.data.append(self.avg_val)

    def get(self):
        return self.data

    def delta(self, start, end):
        try:
            res = self.data[end] - self.data[start]
        except IndexError:
            res = self.data[end]
        return res


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)