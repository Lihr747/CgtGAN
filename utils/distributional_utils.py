import torch
import json
import os
import logging
import torch.distributed as dist
def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

def json_writer(values, f):
    res = []
    for i, g in values:
        res.append({"image_id":str(int(i)), "caption":g})
    json_str = json.dumps(res, indent=4)
    f.write(json_str)

def concat_json_files(jsons, out_json):
    out_list = []
    for j_file in jsons:
        f = open(j_file, 'r')
        result = json.load(f)
        out_list.extend(result)
        f.close()
    # remove redundant caption inference
    out_list_clean = []
    img_key_set = set([])
    for cap in out_list:
        if cap['image_id'] not in img_key_set:
            out_list_clean.append(cap)
            img_key_set.add(cap['image_id'])
    
    json_str = json.dumps(out_list_clean, indent=4)
    f = open(out_json, "w")
    f.write(json_str)

def delete_json_files(jsons):
    for j in jsons:
        if os.path.isfile(j):
            try_delete(j)

def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
    return func_wrapper


@try_once
def try_delete(f):
    os.remove(f)