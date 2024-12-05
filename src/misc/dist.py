"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

by lyuwenyu
"""

import random
import numpy as np 

import mindspore 
import mindspore.nn as nn 
import mindspore.communication # import torch.distributed
import mindspore.communication  as tdist # import torch.distributed as tdist

#from torch.nn.parallel import DistributedDataParallel as DDP

#from torch.utils.data import DistributedSampler
#from torch.utils.data.dataloader import DataLoader
import mindspore.dataset as ds


def init_distributed():
    '''
    distributed setup
    args:
        backend (str), ('nccl', 'gloo')
    '''
    try:
        # # https://pytorch.org/docs/stable/elastic/run.html
        # LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
        # RANK = int(os.getenv('RANK', -1))
        # WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
        
        tdist.init()
        mindspore.communication.comm_func.barrier()

        rank = get_rank()
        #device = torch.device(f'cuda:{rank}')
        #torch.cuda.set_device(device)
        mindspore.context.set_context(device_target="GPU", device_id=rank)
       

        setup_print(rank == 0)
        print('Initialized distributed mode...')

        return True 

    except:
        print('Not init distributed mode.')
        return False 


def setup_print(is_main):
    '''This function disables printing when not in master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not tdist.is_available():
        return False
    if not tdist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return tdist.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return tdist.get_world_size()

    
def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        mindspore.save_checkpoint(*args, **kwargs)



'''def warp_model(model, find_unused_parameters=False, sync_bn=False,):
    if is_dist_available_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model 
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
    return model'''
def warp_model(model, find_unused_parameters=False, sync_bn=False):
    if is_dist_available_and_initialized():
        if sync_bn:
            model = nn.SyncBatchNorm(model)  # 替换同步 BN
        # MindSpore 的分布式训练通过设置 ParallelContext 和 TrainOneStepCell 实现
        mindspore.communication.init("nccl")
        mindspore.context.set_auto_parallel_context(parallel_mode="data_parallel", gradients_mean=True)
    return model



'''def warp_loader(loader, shuffle=False):        
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset, 
                            loader.batch_size, 
                            sampler=sampler, 
                            drop_last=loader.drop_last, 
                            collate_fn=loader.collate_fn, 
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers, )
    return loader'''
def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = ds.DistributedSampler(dataset=loader.dataset, shuffle=shuffle)
        dataset = loader.dataset
        dataset = dataset.use_sampler(sampler)  # 添加分布式采样
        dataset = dataset.batch(loader.batch_size)  # 添加批处理
        loader = dataset.create_dict_iterator()  # 创建迭代器
    return loader




'''def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)'''
def is_parallel(model) -> bool:
    return model.parallel_mode in ["data_parallel", "model_parallel"]



def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


"""def reduce_dict(data, avg=True):
    '''
    Args 
        data dict: input, {k: v, ...}
        avg bool: true
    '''
    world_size = get_world_size()
    if world_size < 2:
        return data
    
    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = mindspore.ops.stack(values,axis=0) #values = torch.stack(values, dim=0)
        tdist.all_reduce(values)

        if avg is True:
            values /= world_size
        
        _data = {k: v for k, v in zip(keys, values)}
    
    return _data"""
def reduce_dict(data, avg=True):
    world_size = get_world_size()
    if world_size < 2:
        return data
    
    keys, values = [], []
    for k in sorted(data.keys()):
        keys.append(k)
        values.append(data[k])

    values = mindspore.ops.stack(values, axis=0)
    values = mindspore.communication.all_reduce(values, op=mindspore.communication.ReduceOp.SUM)

    if avg:
        values /= world_size

    _data = {k: v for k, v in zip(keys, values)}
    return _data




def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    tdist.all_gather_object(data_list, data)
    return data_list

    
import time 
"""def sync_time():
    '''sync_time
    '''
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()"""
def sync_time():
    mindspore.communication.barrier()
    return time.time()




def set_seed(seed):
    # fix the seed for reproducibility
    mindspore.manual_seed(seed) #torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


