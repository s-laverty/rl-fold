#!/usr/bin/env python

'''
This file runs tests to determine hardware capability.

Created on 3/7/2023 by Steven Laverty (lavers@rpi.edu)
'''

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from model.fold_zero_net import FoldZeroNet
from model.fold_q_net import FoldQNet

print('CPU count: {}'.format(mp.cpu_count()))

if torch.cuda.is_available():
    print('CUDA available: {} devices'.format(torch.cuda.device_count()))
else:
    print('CUDA not available')

print('Sharing stragies: {}'.format(mp.get_all_sharing_strategies()))

if dist.is_available():
    print('Distributed available')
    print('GLOO available: {}'.format(dist.is_gloo_available()))
    print('MPI available: {}'.format(dist.is_mpi_available()))
    print('NCCL available: {}'.format(dist.is_nccl_available()))
else:
    print('Distributed not available')


x = torch.arange(24).view(4,3,2)
print(x)
print(x.unfold(0, 2, 1))
print(x.unfold(0, 2, 1)[0, ..., 0])
print(x.unfold(0, 2, 1)[0, ..., 1])

if False:
    x, y, z = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])
    print(x, y, z)
    shared_buf = torch.empty((9), dtype=torch.long)
    torch.cat((x, y, z), out=shared_buf)
    storage = shared_buf.storage()
    x.set_(storage, 0, x.size())
    y.set_(storage, 3, y.size())
    z.set_(storage, 6, y.size())
    print('Packing result (pre-shm)', x, y, z)
    storage.share_memory_()
    print('Packing result (post-shm)', x, y, z)
    print('Tensors are is shared:', x.is_shared(), y.is_shared(), z.is_shared())

    torch.save((x, y, z, storage), 'cap_test.pth')
    del x, y, z, storage
    x, y, z, storage = torch.load('cap_test.pth')
    print('Load result:', x, y, z)
    print('Load result is shared (pre-shm):', x.is_shared(), y.is_shared(), z.is_shared())
    print(storage)
    storage.share_memory_()
    print('Load result is shared (post-shm):', x.is_shared(), y.is_shared(), z.is_shared())

if False:
    net = FoldQNet()
    print(len(list(net.named_parameters())))
    print(len(list(net.parameters())))

if False:
    net = FoldZeroNet()
    adam = optim.Adam(
        net.parameters(),
        betas=[0.5, 0.1]
    )

    optim_params = adam.state_dict()
    print(optim_params)

    adam2 = optim.Adam(
        net.parameters()
    )
    print(adam2.state_dict())

    adam2.load_state_dict(optim_params)
    print(adam2.state_dict())


    scheduler = optim.lr_scheduler.MultiStepLR(adam, [1, 6, 9], gamma=0.6)

    adam.step()
    scheduler.step()
    adam.step()
    scheduler.step()

    scheduler_params = scheduler.state_dict()
    print(scheduler_params)

    scheduler2 = optim.lr_scheduler.MultiStepLR(adam, [], gamma=0.1)
    print(scheduler2.state_dict())
    scheduler2.load_state_dict(scheduler_params)
    print(scheduler2.state_dict())
