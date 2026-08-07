#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gc
from fractions import Fraction
import argparse

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("--arg_main_path",type=str); parser.add_argument("--arg_BS",type=int)
parser.add_argument("--arg_epochs",type=float)
parser.add_argument("--arg_lr",type=float)
parser.add_argument("--arg_num_layers_gnn",type=int); parser.add_argument("--arg_chan_gnn",type=int)

args = parser.parse_args()

thepath = args.arg_main_path
thepath_data = thepath+'Data/'
thepath_model = thepath+'Model/'
thepath_analysis = thepath+'Analysis/'


dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
world_size = dist.get_world_size()
dist_rank = dist.get_rank()

if dist_rank == 0:
    os.makedirs(thepath_model, exist_ok=True)
    os.makedirs(thepath_analysis, exist_ok=True)
dist.barrier()


exec(open('Various_Functions.py').read())

thenetwork0 = np.loadtxt(thepath_data+'my_net_XRB.dat').astype('int')
thenetwork0.shape
my_network = np.zeros((20000,2)) # N, Z
d=0
for i in range(len(thenetwork0)):
    for j in range(thenetwork0[i,1],thenetwork0[i,2]+1):
        my_network[d,0] = j
        my_network[d,1] = thenetwork0[i,0]
        d+=1
my_network = my_network[:d]
net_length = len(my_network)

themaxN = int(np.max(my_network[:,0])+1)
themaxZ = int(np.max(my_network[:,1])+1)

my_network_map = network_map_converter(subtract_factor=0)


# Small preprocessing metadata
metadata = np.load(thepath_data+f"Training_Data/metadata.npz")

N_cut = int(metadata["N_cut"])
Z_cut = int(metadata["Z_cut"])

FiLM_channels_len = int(metadata["FiLM_channels_len"])
in_channels_iso_len = int(metadata["in_channels_iso_len"])
in_channels_rea_len = int(metadata["in_channels_rea_len"])

iso_len = int(metadata["iso_len"])
reaction_len = int(metadata["reaction_len"])
num_rate_types = int(metadata["num_rate_types"])
net_length = int(metadata["net_length"])
num_classes = int(metadata["num_classes"])

the_clip_X_in = float(metadata["clip_X_in"])
the_log_subtract_factor = float(metadata["log_subtract_factor"])

cut1_val = float(metadata["cut1_val"])
cut2_val = float(metadata["cut2_val"])


iso_NZ = np.load(thepath_data+f'Training_Data/iso_NZ.npy')
r_type = np.load(thepath_data+f'Training_Data/r_type.npy')
i2r_src = np.load(thepath_data+f'Training_Data/i2r_src.npy')
i2r_dst = np.load(thepath_data+f'Training_Data/i2r_dst.npy')
r2i_src = np.load(thepath_data+f'Training_Data/r2i_src.npy')
r2i_dst = np.load(thepath_data+f'Training_Data/r2i_dst.npy')
r2i_coeff = np.load(thepath_data+f'Training_Data/r2i_coeff.npy')
i2i_src = np.load(thepath_data+f'Training_Data/i2i_src.npy')
i2i_dst = np.load(thepath_data+f'Training_Data/i2i_dst.npy')

slicing_rea = np.load(thepath_data+f'Training_Data/slicing_rea.npy')
slicing_i2r = np.load(thepath_data+f'Training_Data/slicing_i2r.npy')
slicing_r2i = np.load(thepath_data+f'Training_Data/slicing_r2i.npy')
slicing_i2i = np.load(thepath_data+f'Training_Data/slicing_i2i.npy')

iso_NZ = torch.tensor(iso_NZ).to(device)
r_type = torch.tensor(r_type).to(device)
i2r_src = torch.tensor(i2r_src).to(device)
i2r_dst = torch.tensor(i2r_dst).to(device)
r2i_src = torch.tensor(r2i_src).to(device)
r2i_dst = torch.tensor(r2i_dst).to(device)
r2i_coeff = torch.tensor(r2i_coeff).to(device).to(torch.float32)
i2i_src = torch.tensor(i2i_src).to(device)
i2i_dst = torch.tensor(i2i_dst).to(device)


len_Tr_type = np.load(thepath_data+f'Training_Data/len_Tr_type.npy')
len_Va_type = np.load(thepath_data+f'Training_Data/len_Va_type.npy')
len_Te_type = np.load(thepath_data+f'Training_Data/len_Te_type.npy')

len_cut1_cut_Tr_type = np.load(thepath_data+f'Training_Data/len_cut1_cut_Tr_type.npy')
len_cut1_cut_Va_type = np.load(thepath_data+f'Training_Data/len_cut1_cut_Va_type.npy')
len_cut1_cut_Te_type = np.load(thepath_data+f'Training_Data/len_cut1_cut_Te_type.npy')

len_cut2_cut_Tr_type = np.load(thepath_data+f'Training_Data/len_cut2_cut_Tr_type.npy')
len_cut2_cut_Va_type = np.load(thepath_data+f'Training_Data/len_cut2_cut_Va_type.npy')
len_cut2_cut_Te_type = np.load(thepath_data+f'Training_Data/len_cut2_cut_Te_type.npy')

len_cut1_cut_Tr = np.load(thepath_data+f'Training_Data/len_cut1_cut_Tr.npy')[0]
len_cut1_cut_Va = np.load(thepath_data+f'Training_Data/len_cut1_cut_Va.npy')[0]
len_cut1_cut_Te = np.load(thepath_data+f'Training_Data/len_cut1_cut_Te.npy')[0]

len_cut2_cut_Tr = np.load(thepath_data+f'Training_Data/len_cut2_cut_Tr.npy')[0]
len_cut2_cut_Va = np.load(thepath_data+f'Training_Data/len_cut2_cut_Va.npy')[0]
len_cut2_cut_Te = np.load(thepath_data+f'Training_Data/len_cut2_cut_Te.npy')[0]

len_Tr_true_classes = np.load(thepath_data+f'Training_Data/len_Tr_true_classes.npy')
len_Va_true_classes = np.load(thepath_data+f'Training_Data/len_Va_true_classes.npy')
len_Te_true_classes = np.load(thepath_data+f'Training_Data/len_Te_true_classes.npy')
len_Tr_type_true_classes = np.load(thepath_data+f'Training_Data/len_Tr_type_true_classes.npy')
len_Va_type_true_classes = np.load(thepath_data+f'Training_Data/len_Va_type_true_classes.npy')
len_Te_type_true_classes = np.load(thepath_data+f'Training_Data/len_Te_type_true_classes.npy')


Tr_length = np.sum(len_Tr_type)
Va_length = np.sum(len_Va_type)
Te_length = np.sum(len_Te_type)


BS = args.arg_BS
print("the number of training batches per GPU:", Tr_length//BS+1)

Loss_RG = nn.SmoothL1Loss(reduction='none',beta=0.001)
    

exec(open('GNN_Architecture.py').read())

class my_model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Net = PureGNN_Final()

    def forward(self, input_i, input_r):

        out = self.Net(input_i, input_r)[:,:,0]
        
        return out

param_dic_FiLM = {'num_lin':4, 'lin1':128,'lin2':256,'lin3':512,'lin4':0}

param_dic_GNN = {'num_layers':args.arg_num_layers_gnn, 'chan':args.arg_chan_gnn}



############## Training STAGE ##############
############## Training STAGE ##############
############## Training STAGE ##############

start = time.time()

epochs = int(args.arg_epochs)#50000

thepath_analysis_config = thepath_analysis

for theset in ['Tr','Va']:
    globals()[f'loss_{theset}'] = np.zeros(epochs)
    
    globals()[f'err_{theset}'] = np.zeros(epochs); globals()[f'err_sum_{theset}'] = np.zeros(epochs); globals()[f'err_tot_{theset}'] = np.zeros(epochs)
    
    globals()[f'err_cut1_{theset}'] = np.zeros(epochs); globals()[f'err_cut2_{theset}'] = np.zeros(epochs)
    
    globals()[f'acc_{theset}'] = np.zeros(epochs)
    
    for type_i in range(3):
        globals()[f'err_{theset}_{type_i}'] = np.zeros(epochs); globals()[f'err_sum_{theset}_{type_i}'] = np.zeros(epochs)
                
        globals()[f'err_cut1_{theset}_{type_i}'] = np.zeros(epochs); globals()[f'err_cut2_{theset}_{type_i}'] = np.zeros(epochs)

        globals()[f'acc_{theset}_{type_i}'] = np.zeros(epochs)
        
        for c_i in range(num_classes):
            globals()[f'precis_{c_i}_{theset}_{type_i}'] = np.zeros(epochs)
            globals()[f'recall_{c_i}_{theset}_{type_i}'] = np.zeros(epochs)

    for c_i in range(num_classes):
        globals()[f'precis_{c_i}_{theset}'] = np.zeros(epochs)
        globals()[f'recall_{c_i}_{theset}'] = np.zeros(epochs)



    
model = my_model().to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)
    
save_path = thepath_model
print("save_path:", save_path)
    
print("thepath_analysis_config:", thepath_analysis_config)
    
optimizer = optim.Adam(model.parameters(), lr=args.arg_lr)
    
train_ds = NumpyMemMap_G(thepath_data+f'Training_Data/in_Tr_1.npy', thepath_data+f'Training_Data/in_Tr_2.npy', thepath_data+f'Training_Data/in_Tr_3.npy',
                         thepath_data+f'Training_Data/out_Tr.npy', thepath_data+f'Training_Data/out_w_Tr.npy', 
                         thepath_data+f'Training_Data/out_cl_Tr.npy',
                         thepath_data+f'Training_Data/out_ref_Tr.npy',thepath_data+f'Training_Data/out_og_ref_Tr.npy')
val_ds = NumpyMemMap_G(thepath_data+f'Training_Data/in_Va_1.npy', thepath_data+f'Training_Data/in_Va_2.npy', thepath_data+f'Training_Data/in_Va_3.npy',
                       thepath_data+f'Training_Data/out_Va.npy',thepath_data+f'Training_Data/out_w_Va.npy',
                       thepath_data+f'Training_Data/out_cl_Va.npy',
                       thepath_data+f'Training_Data/out_ref_Va.npy',thepath_data+f'Training_Data/out_og_ref_Va.npy')
    
# Shard across GPUs
train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=dist_rank, shuffle=True, seed=0, drop_last=True)
val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=dist_rank, shuffle=False, seed=0, drop_last=False)
    
# Per-GPU micro-batch size = your BS
train_loader = DataLoader(train_ds, batch_size=BS, sampler=train_sampler, num_workers=2, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BS, sampler=val_sampler, num_workers=2, pin_memory=True, persistent_workers=True)
    
for epoch in range(epochs):
    train_sampler.set_epoch(epoch)
        
    theloss = torch.zeros(1,device=device)
    err = torch.zeros(1,device=device)
    err_cut1 = torch.zeros(1,device=device)
    err_cut2 = torch.zeros(1,device=device)
    err_sum = torch.zeros(1,device=device)
        
    for type_i in range(3):
        globals()[f'err_{type_i}'] = torch.zeros(1,device=device)
        globals()[f'err_cut1_{type_i}'] = torch.zeros(1,device=device)
        globals()[f'err_cut2_{type_i}'] = torch.zeros(1,device=device)
        globals()[f'err_sum_{type_i}'] = torch.zeros(1,device=device)
            
    err_tot = torch.zeros(1,device=device)

    len_Tr_pred_classes = torch.zeros(num_classes,device=device)
    len_Tr_type_pred_classes = torch.zeros((3,num_classes),device=device)

    acc = torch.zeros(1,device=device)
    for c_i in range(num_classes):
        globals()[f'precis_{c_i}'] = torch.zeros(1,device=device)
        globals()[f'recall_{c_i}'] = torch.zeros(1,device=device)
    for type_i in range(3):
        globals()[f'acc_{type_i}'] = torch.zeros(1,device=device)
        for c_i in range(num_classes):
            globals()[f'precis_{c_i}_{type_i}'] = torch.zeros(1,device=device)
            globals()[f'recall_{c_i}_{type_i}'] = torch.zeros(1,device=device)

    model.train()
    for theinput_1, theinput_2, theinput_3, thelabel_rg, thelabel_w_rg, thelabel_cl, out_ref_b, out_og_ref_b in train_loader:
        thelabel_cl_ref = thelabel_cl.numpy()
        out_ref_b = out_ref_b.numpy(); out_og_ref_b = out_og_ref_b.numpy()
        
        optimizer.zero_grad()
        
        theinput_2 = theinput_2.to(device, non_blocking=True)
        theinput_3 = theinput_3.to(device, non_blocking=True)
        thelabel_rg = thelabel_rg.to(device, non_blocking=True)
        thelabel_w_rg = thelabel_w_rg.to(device, non_blocking=True)
        
        thepred_rg = model(theinput_2,theinput_3)

        Loss = torch.sum(Loss_RG(thepred_rg,thelabel_rg)*thelabel_w_rg)/net_length/len(thelabel_rg)

        Loss.backward()
        
        optimizer.step()

        theloss += float(Loss.item())*len(thelabel_rg)/Tr_length
        
        theinput_ref = theinput_2[:,:,0].cpu().detach().numpy()

        theinput_ref = 10**((theinput_ref-1)*abs(np.log10(the_clip_X_in)/2))
        
        thepred_ref = thepred_rg.cpu().detach().to(torch.float64).numpy()*the_log_subtract_factor

        thepred_delx = np.sign(thepred_ref.copy())*10**(abs(thepred_ref.copy())-17)*theinput_ref
        
        thepred_ref2 = thepred_delx + theinput_ref
            
        thepred_ref2[thepred_ref2<=the_clip_X_in] = the_clip_X_in
        thepred_ref2 = np.log10(thepred_ref2)

        thepred_cl_ref = np.sign(thepred_ref)
        thepred_cl_ref[abs(thepred_ref)<0.05] = 0
        thepred_cl_ref[thepred_cl_ref==-1] = 2
            
        thepred_ref_net = thepred_ref.copy(); thepred_ref2_net = thepred_ref2.copy()
        out_ref_b_net = out_ref_b.copy(); out_og_ref_b_net = out_og_ref_b.copy()
        err += Error_RG(thepred_ref_net, out_ref_b_net)[0]
        err_cut1 += Error_RG(thepred_ref_net[abs(out_ref_b_net)>cut1_val], out_ref_b_net[abs(out_ref_b_net)>cut1_val])[0]
        err_cut2 += Error_RG(thepred_ref_net[abs(out_ref_b_net)>cut2_val], out_ref_b_net[abs(out_ref_b_net)>cut2_val])[0]
        err_sum += Error_RG(thepred_ref2_net, out_og_ref_b_net)[0]

        for type_i in range(3):
            theindices = np.where(theinput_1[:,-1].numpy().astype('int')==type_i)
            thepred_ref_net = thepred_ref[theindices]; thepred_ref2_net = thepred_ref2[theindices]
            out_ref_b_net = out_ref_b[theindices]; out_og_ref_b_net = out_og_ref_b[theindices]
            globals()[f'err_{type_i}'] += Error_RG(thepred_ref_net, out_ref_b_net)[0]
            globals()[f'err_cut1_{type_i}'] += Error_RG(thepred_ref_net[abs(out_ref_b_net)>cut1_val], out_ref_b_net[abs(out_ref_b_net)>cut1_val])[0]
            globals()[f'err_cut2_{type_i}'] += Error_RG(thepred_ref_net[abs(out_ref_b_net)>cut2_val], out_ref_b_net[abs(out_ref_b_net)>cut2_val])[0]
            globals()[f'err_sum_{type_i}'] += Error_RG(thepred_ref2_net, out_og_ref_b_net)[0]
        
        err_tot += Error_RG(thepred_ref, out_ref_b)[0]

        acc += len(np.where(thepred_cl_ref==thelabel_cl_ref)[0])
        for c_i in range(num_classes):
            pred_is = np.where(thepred_cl_ref==c_i)
            true_is = np.where(thelabel_cl_ref==c_i)
            globals()[f'precis_{c_i}'] += len(np.where(thepred_cl_ref[pred_is]==thelabel_cl_ref[pred_is])[0])
            globals()[f'recall_{c_i}'] += len(np.where(thepred_cl_ref[true_is]==thelabel_cl_ref[true_is])[0])
            len_Tr_pred_classes[c_i] += len(pred_is[0])
        for type_i in range(3):
            theindices = np.where(theinput_1[:,-1].numpy().astype('int')==type_i)
            globals()[f'acc_{type_i}'] += len(np.where(thepred_cl_ref[theindices]==thelabel_cl_ref[theindices])[0])
            for c_i in range(num_classes):
                pred_is = np.where(thepred_cl_ref[theindices]==c_i)
                true_is = np.where(thelabel_cl_ref[theindices]==c_i)
                globals()[f'precis_{c_i}_{type_i}'] += len(np.where(thepred_cl_ref[theindices][pred_is]==thelabel_cl_ref[theindices][pred_is])[0])
                globals()[f'recall_{c_i}_{type_i}'] += len(np.where(thepred_cl_ref[theindices][true_is]==thelabel_cl_ref[theindices][true_is])[0])
                len_Tr_type_pred_classes[type_i,c_i] += len(pred_is[0])

        del theinput_1, theinput_2, theinput_3, thelabel_rg, thepred_rg, Loss
    
    torch.distributed.reduce(theloss, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err_cut1, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err_cut2, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err_tot, dst=0, op=torch.distributed.ReduceOp.SUM)

    for type_i in range(3):
        torch.distributed.reduce(globals()[f'err_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(globals()[f'err_cut1_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(globals()[f'err_cut2_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(globals()[f'err_sum_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)

    torch.distributed.reduce(len_Tr_pred_classes, dst=0, op=torch.distributed.ReduceOp.SUM)  # only rank 0 receives the sums
    torch.distributed.reduce(len_Tr_type_pred_classes, dst=0, op=torch.distributed.ReduceOp.SUM)  # only rank 0 receives the sums

    torch.distributed.reduce(acc, dst=0, op=torch.distributed.ReduceOp.SUM)
    for c_i in range(num_classes):
        torch.distributed.reduce(globals()[f'precis_{c_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(globals()[f'recall_{c_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
    for type_i in range(3):
        torch.distributed.reduce(globals()[f'acc_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        for c_i in range(num_classes):
            torch.distributed.reduce(globals()[f'precis_{c_i}_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(globals()[f'recall_{c_i}_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
    
    if dist_rank == 0:
        loss_Tr[epoch] = float(theloss.item())
        err_Tr[epoch] = float(err.item())
        err_cut1_Tr[epoch] = float(err_cut1.item())
        err_cut2_Tr[epoch] = float(err_cut2.item())
        err_sum_Tr[epoch] = float(err_sum.item())
        err_tot_Tr[epoch] = float(err_tot.item())
        
        for type_i in range(3):
            globals()[f'err_Tr_{type_i}'][epoch] = float(globals()[f'err_{type_i}'].item())
            globals()[f'err_cut1_Tr_{type_i}'][epoch] = float(globals()[f'err_cut1_{type_i}'].item())
            globals()[f'err_cut2_Tr_{type_i}'][epoch] = float(globals()[f'err_cut2_{type_i}'].item())
            globals()[f'err_sum_Tr_{type_i}'][epoch] = float(globals()[f'err_sum_{type_i}'].item())
            
        acc_Tr[epoch] = float(acc.item())
        for c_i in range(num_classes):
            globals()[f'precis_{c_i}_Tr'][epoch] = float(globals()[f'precis_{c_i}'].item())
            globals()[f'recall_{c_i}_Tr'][epoch] = float(globals()[f'recall_{c_i}'].item())
        for type_i in range(3):
            globals()[f'acc_Tr_{type_i}'][epoch] = float(globals()[f'acc_{type_i}'].item())
            for c_i in range(num_classes):
                globals()[f'precis_{c_i}_Tr_{type_i}'][epoch] = float(globals()[f'precis_{c_i}_{type_i}'].item())
                globals()[f'recall_{c_i}_Tr_{type_i}'][epoch] = float(globals()[f'recall_{c_i}_{type_i}'].item())

            
    theloss = torch.zeros(1,device=device)
    err = torch.zeros(1,device=device)
    err_cut1 = torch.zeros(1,device=device)
    err_cut2 = torch.zeros(1,device=device)
    err_sum = torch.zeros(1,device=device)
    
    for type_i in range(3):
        globals()[f'err_{type_i}'] = torch.zeros(1,device=device)
        globals()[f'err_cut1_{type_i}'] = torch.zeros(1,device=device)
        globals()[f'err_cut2_{type_i}'] = torch.zeros(1,device=device)
        globals()[f'err_sum_{type_i}'] = torch.zeros(1,device=device)
        
    err_tot = torch.zeros(1,device=device)

    len_Va_pred_classes = torch.zeros(num_classes,device=device)
    len_Va_type_pred_classes = torch.zeros((3,num_classes),device=device)

    acc = torch.zeros(1,device=device)
    for c_i in range(num_classes):
        globals()[f'precis_{c_i}'] = torch.zeros(1,device=device)
        globals()[f'recall_{c_i}'] = torch.zeros(1,device=device)
    for type_i in range(3):
        globals()[f'acc_{type_i}'] = torch.zeros(1,device=device)
        for c_i in range(num_classes):
            globals()[f'precis_{c_i}_{type_i}'] = torch.zeros(1,device=device)
            globals()[f'recall_{c_i}_{type_i}'] = torch.zeros(1,device=device)
        
    model.eval()
  #  enable_dropout(model)
    with torch.no_grad():
        for theinput_1, theinput_2, theinput_3, thelabel_rg, thelabel_w_rg, thelabel_cl, out_ref_b, out_og_ref_b in val_loader:
            thelabel_cl_ref = thelabel_cl.numpy()
            out_ref_b = out_ref_b.numpy(); out_og_ref_b = out_og_ref_b.numpy()
            
            theinput_2 = theinput_2.to(device, non_blocking=True)
            theinput_3 = theinput_3.to(device, non_blocking=True)
            thelabel_rg = thelabel_rg.to(device, non_blocking=True)
            thelabel_w_rg = thelabel_w_rg.to(device, non_blocking=True)
            
            thepred_rg = model(theinput_2,theinput_3)
            
            Loss = torch.sum(Loss_RG(thepred_rg,thelabel_rg)*thelabel_w_rg)/net_length/len(thelabel_rg)
            
            theloss += float(Loss.item())*len(thelabel_rg)/Va_length

            theinput_ref = theinput_2[:,:,0].cpu().detach().numpy()
                
            theinput_ref = 10**((theinput_ref-1)*abs(np.log10(the_clip_X_in)/2))

            thepred_ref = thepred_rg.cpu().detach().to(torch.float64).numpy()*the_log_subtract_factor
                
            thepred_delx = np.sign(thepred_ref.copy())*10**(abs(thepred_ref.copy())-17)*theinput_ref
            
            thepred_ref2 = thepred_delx + theinput_ref
                
            thepred_ref2[thepred_ref2<=the_clip_X_in] = the_clip_X_in
            thepred_ref2 = np.log10(thepred_ref2)

            thepred_cl_ref = np.sign(thepred_ref)
            thepred_cl_ref[abs(thepred_ref)<0.05] = 0
            thepred_cl_ref[thepred_cl_ref==-1] = 2
            
            thepred_ref_net = thepred_ref.copy(); thepred_ref2_net = thepred_ref2.copy()
            out_ref_b_net = out_ref_b.copy(); out_og_ref_b_net = out_og_ref_b.copy()
            err += Error_RG(thepred_ref_net, out_ref_b_net)[0]
            err_cut1 += Error_RG(thepred_ref_net[abs(out_ref_b_net)>cut1_val], out_ref_b_net[abs(out_ref_b_net)>cut1_val])[0]
            err_cut2 += Error_RG(thepred_ref_net[abs(out_ref_b_net)>cut2_val], out_ref_b_net[abs(out_ref_b_net)>cut2_val])[0]
            err_sum += Error_RG(thepred_ref2_net, out_og_ref_b_net)[0]

            for type_i in range(3):
                theindices = np.where(theinput_1[:,-1].numpy().astype('int')==type_i)
                thepred_ref_net = thepred_ref[theindices]; thepred_ref2_net = thepred_ref2[theindices]
                out_ref_b_net = out_ref_b[theindices]; out_og_ref_b_net = out_og_ref_b[theindices]
                globals()[f'err_{type_i}'] += Error_RG(thepred_ref_net, out_ref_b_net)[0]
                globals()[f'err_cut1_{type_i}'] += Error_RG(thepred_ref_net[abs(out_ref_b_net)>cut1_val], out_ref_b_net[abs(out_ref_b_net)>cut1_val])[0]
                globals()[f'err_cut2_{type_i}'] += Error_RG(thepred_ref_net[abs(out_ref_b_net)>cut2_val], out_ref_b_net[abs(out_ref_b_net)>cut2_val])[0]
                globals()[f'err_sum_{type_i}'] += Error_RG(thepred_ref2_net, out_og_ref_b_net)[0]
            
            err_tot += Error_RG(thepred_ref, out_ref_b)[0]

            acc += len(np.where(thepred_cl_ref==thelabel_cl_ref)[0])
            for c_i in range(num_classes):
                pred_is = np.where(thepred_cl_ref==c_i)
                true_is = np.where(thelabel_cl_ref==c_i)
                globals()[f'precis_{c_i}'] += len(np.where(thepred_cl_ref[pred_is]==thelabel_cl_ref[pred_is])[0])
                globals()[f'recall_{c_i}'] += len(np.where(thepred_cl_ref[true_is]==thelabel_cl_ref[true_is])[0])
                len_Va_pred_classes[c_i] += len(pred_is[0])
            for type_i in range(3):
                theindices = np.where(theinput_1[:,-1].numpy().astype('int')==type_i)
                globals()[f'acc_{type_i}'] += len(np.where(thepred_cl_ref[theindices]==thelabel_cl_ref[theindices])[0])
                for c_i in range(num_classes):
                    pred_is = np.where(thepred_cl_ref[theindices]==c_i)
                    true_is = np.where(thelabel_cl_ref[theindices]==c_i)
                    globals()[f'precis_{c_i}_{type_i}'] += len(np.where(thepred_cl_ref[theindices][pred_is]==thelabel_cl_ref[theindices][pred_is])[0])
                    globals()[f'recall_{c_i}_{type_i}'] += len(np.where(thepred_cl_ref[theindices][true_is]==thelabel_cl_ref[theindices][true_is])[0])
                    len_Va_type_pred_classes[type_i,c_i] += len(pred_is[0])

            del theinput_1, theinput_2, theinput_3, thelabel_rg, thepred_rg, Loss

    torch.distributed.reduce(theloss, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err_cut1, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err_cut2, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(err_tot, dst=0, op=torch.distributed.ReduceOp.SUM)

    for type_i in range(3):
        torch.distributed.reduce(globals()[f'err_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(globals()[f'err_cut1_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(globals()[f'err_cut2_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(globals()[f'err_sum_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        
    torch.distributed.reduce(len_Va_pred_classes, dst=0, op=torch.distributed.ReduceOp.SUM)  # only rank 0 receives the sums
    torch.distributed.reduce(len_Va_type_pred_classes, dst=0, op=torch.distributed.ReduceOp.SUM)  # only rank 0 receives the sums

    torch.distributed.reduce(acc, dst=0, op=torch.distributed.ReduceOp.SUM)
    for c_i in range(num_classes):
        torch.distributed.reduce(globals()[f'precis_{c_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(globals()[f'recall_{c_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
    for type_i in range(3):
        torch.distributed.reduce(globals()[f'acc_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
        for c_i in range(num_classes):
            torch.distributed.reduce(globals()[f'precis_{c_i}_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(globals()[f'recall_{c_i}_{type_i}'], dst=0, op=torch.distributed.ReduceOp.SUM)
            
    if dist_rank == 0:
        loss_Va[epoch] = float(theloss.item())
        err_Va[epoch] = float(err.item())
        err_cut1_Va[epoch] = float(err_cut1.item())
        err_cut2_Va[epoch] = float(err_cut2.item())
        err_sum_Va[epoch] = float(err_sum.item())
        err_tot_Va[epoch] = float(err_tot.item())
        
        for type_i in range(3):
            globals()[f'err_Va_{type_i}'][epoch] = float(globals()[f'err_{type_i}'].item())
            globals()[f'err_cut1_Va_{type_i}'][epoch] = float(globals()[f'err_cut1_{type_i}'].item())
            globals()[f'err_cut2_Va_{type_i}'][epoch] = float(globals()[f'err_cut2_{type_i}'].item())
            globals()[f'err_sum_Va_{type_i}'][epoch] = float(globals()[f'err_sum_{type_i}'].item())

        acc_Va[epoch] = float(acc.item())
        for c_i in range(num_classes):
            globals()[f'precis_{c_i}_Va'][epoch] = float(globals()[f'precis_{c_i}'].item())
            globals()[f'recall_{c_i}_Va'][epoch] = float(globals()[f'recall_{c_i}'].item())
        for type_i in range(3):
            globals()[f'acc_Va_{type_i}'][epoch] = float(globals()[f'acc_{type_i}'].item())
            for c_i in range(num_classes):
                globals()[f'precis_{c_i}_Va_{type_i}'][epoch] = float(globals()[f'precis_{c_i}_{type_i}'].item())
                globals()[f'recall_{c_i}_Va_{type_i}'][epoch] = float(globals()[f'recall_{c_i}_{type_i}'].item())

        err_Tr[epoch] = err_Tr[epoch]/net_length/Tr_length
        err_cut1_Tr[epoch] = err_cut1_Tr[epoch]/len_cut1_cut_Tr
        err_cut2_Tr[epoch] = err_cut2_Tr[epoch]/len_cut2_cut_Tr
        err_Va[epoch] = err_Va[epoch]/net_length/Va_length
        err_cut1_Va[epoch] = err_cut1_Va[epoch]/len_cut1_cut_Va
        err_cut2_Va[epoch] = err_cut2_Va[epoch]/len_cut2_cut_Va
        
        err_sum_Tr[epoch] = err_sum_Tr[epoch]/net_length/Tr_length
        err_sum_Va[epoch] = err_sum_Va[epoch]/net_length/Va_length
        
        for type_i in range(3):
            globals()[f'err_Tr_{type_i}'][epoch] = globals()[f'err_Tr_{type_i}'][epoch]/net_length/len_Tr_type[type_i]
            globals()[f'err_cut1_Tr_{type_i}'][epoch] = globals()[f'err_cut1_Tr_{type_i}'][epoch]/len_cut1_cut_Tr_type[type_i]
            globals()[f'err_cut2_Tr_{type_i}'][epoch] = globals()[f'err_cut2_Tr_{type_i}'][epoch]/len_cut2_cut_Tr_type[type_i]
            globals()[f'err_Va_{type_i}'][epoch] = globals()[f'err_Va_{type_i}'][epoch]/net_length/len_Va_type[type_i]
            globals()[f'err_cut1_Va_{type_i}'][epoch] = globals()[f'err_cut1_Va_{type_i}'][epoch]/len_cut1_cut_Va_type[type_i]
            globals()[f'err_cut2_Va_{type_i}'][epoch] = globals()[f'err_cut2_Va_{type_i}'][epoch]/len_cut2_cut_Va_type[type_i]
            
            globals()[f'err_sum_Tr_{type_i}'][epoch] = globals()[f'err_sum_Tr_{type_i}'][epoch]/net_length/len_Tr_type[type_i]
            globals()[f'err_sum_Va_{type_i}'][epoch] = globals()[f'err_sum_Va_{type_i}'][epoch]/net_length/len_Va_type[type_i]
                        
        err_tot_Tr[epoch] = err_tot_Tr[epoch]/(N_cut*Z_cut)/Tr_length
        err_tot_Va[epoch] = err_tot_Va[epoch]/(N_cut*Z_cut)/Va_length

        acc_Tr[epoch] = acc_Tr[epoch]/Tr_length/net_length
        acc_Va[epoch] = acc_Va[epoch]/Va_length/net_length
        for c_i in range(num_classes):
            globals()[f'precis_{c_i}_Tr'][epoch] = globals()[f'precis_{c_i}_Tr'][epoch]/float(len_Tr_pred_classes[c_i].item())
            globals()[f'recall_{c_i}_Tr'][epoch] = globals()[f'recall_{c_i}_Tr'][epoch]/float(len_Tr_true_classes[c_i].item())
            globals()[f'precis_{c_i}_Va'][epoch] = globals()[f'precis_{c_i}_Va'][epoch]/float(len_Va_pred_classes[c_i].item())
            globals()[f'recall_{c_i}_Va'][epoch] = globals()[f'recall_{c_i}_Va'][epoch]/float(len_Va_true_classes[c_i].item())
            
        for type_i in range(3):
            globals()[f'acc_Tr_{type_i}'][epoch] = globals()[f'acc_Tr_{type_i}'][epoch]/len_Tr_type[type_i]/net_length
            globals()[f'acc_Va_{type_i}'][epoch] = globals()[f'acc_Va_{type_i}'][epoch]/len_Va_type[type_i]/net_length
            for c_i in range(num_classes):
                globals()[f'precis_{c_i}_Tr_{type_i}'][epoch] = globals()[f'precis_{c_i}_Tr_{type_i}'][epoch]/float(len_Tr_type_pred_classes[type_i,c_i].item())
                globals()[f'recall_{c_i}_Tr_{type_i}'][epoch] = globals()[f'recall_{c_i}_Tr_{type_i}'][epoch]/float(len_Tr_type_true_classes[type_i,c_i].item())
                globals()[f'precis_{c_i}_Va_{type_i}'][epoch] = globals()[f'precis_{c_i}_Va_{type_i}'][epoch]/float(len_Va_type_pred_classes[type_i,c_i].item())
                globals()[f'recall_{c_i}_Va_{type_i}'][epoch] = globals()[f'recall_{c_i}_Va_{type_i}'][epoch]/float(len_Va_type_true_classes[type_i,c_i].item())
                
        if (epoch+1)%1==0:
            print('Epoch:', '%05d' % (epoch + 1), "time: ", timeSince(start), "lr: ", optimizer.param_groups[0]['lr'], '\n',
                  'loss_Tr      =', '{:.3e}'.format(loss_Tr[epoch]), 'loss_Va      =', '{:.3e}'.format(loss_Va[epoch]), '\n',
                  'err_Tr       =', '{:.3e}'.format(err_Tr[epoch]), 'err_Va       =', '{:.3e}'.format(err_Va[epoch]), 
                  'err_sum_Tr   =', '{:.3e}'.format(err_sum_Tr[epoch]), 'err_sum_Va   =', '{:.3e}'.format(err_sum_Va[epoch]), '\n',
                  
                  'err_cut1_Tr  =', '{:.3e}'.format(err_cut1_Tr[epoch]), 'err_cut1_Va  =', '{:.3e}'.format(err_cut1_Va[epoch]), '\n',
                  'err_cut1_Tr_0=', '{:.3e}'.format(err_cut1_Tr_0[epoch]), 'err_cut1_Tr_1=', '{:.3e}'.format(err_cut1_Tr_1[epoch]),'err_cut1_Tr_2=', '{:.3e}'.format(err_cut1_Tr_2[epoch]), '\n', 
                  'err_cut1_Va_0=', '{:.3e}'.format(err_cut1_Va_0[epoch]), 'err_cut1_Va_1=', '{:.3e}'.format(err_cut1_Va_1[epoch]), 'err_cut1_Va_2=', '{:.3e}'.format(err_cut1_Va_2[epoch]), '\n',
                  
                  'acc_Tr       =', '{:.3e}'.format(acc_Tr[epoch]), 'acc_Va       =', '{:.3e}'.format(acc_Va[epoch]), '\n', 
                  
                  'precis_0_Tr  =', '{:.3e}'.format(precis_0_Tr[epoch]), 'recall_0_Tr  =', '{:.3e}'.format(recall_0_Tr[epoch]), 
                  'precis_0_Va  =', '{:.3e}'.format(precis_0_Va[epoch]), 'recall_0_Va  =', '{:.3e}'.format(recall_0_Va[epoch]), '\n', 
                  'precis_1_Tr  =', '{:.3e}'.format(precis_1_Tr[epoch]), 'recall_1_Tr  =', '{:.3e}'.format(recall_1_Tr[epoch]), 
                  'precis_1_Va  =', '{:.3e}'.format(precis_1_Va[epoch]), 'recall_1_Va  =', '{:.3e}'.format(recall_1_Va[epoch]), '\n', 
                  'precis_2_Tr  =', '{:.3e}'.format(precis_2_Tr[epoch]), 'recall_2_Tr  =', '{:.3e}'.format(recall_2_Tr[epoch]), 
                  'precis_2_Va  =', '{:.3e}'.format(precis_2_Va[epoch]), 'recall_2_Va  =', '{:.3e}'.format(recall_2_Va[epoch]), '\n',
                  
                  'acc_Tr_0     =', '{:.3e}'.format(acc_Tr_0[epoch]), 'acc_Tr_1     =', '{:.3e}'.format(acc_Tr_1[epoch]), 'acc_Tr_2     =', '{:.3e}'.format(acc_Tr_2[epoch]), '\n',
                  'acc_Va_0     =', '{:.3e}'.format(acc_Va_0[epoch]), 'acc_Va_1     =', '{:.3e}'.format(acc_Va_1[epoch]), 'acc_Va_2     =', '{:.3e}'.format(acc_Va_2[epoch]),
                 )

        if epoch>1:
            if loss_Va[epoch]<np.min(loss_Va[:epoch]) or err_Va[epoch]<np.min(err_Va[:epoch]) or err_Va_1[epoch]<np.min(err_Va_1[:epoch]) or err_cut1_Va[epoch]<np.min(err_cut1_Va[:epoch]) or err_cut1_Va_1[epoch]<np.min(err_cut1_Va_1[:epoch]):
                print("min approached, saving model...")
                torch.save(model.module.state_dict(), save_path+f"{epoch}.pt")

        if (epoch+1)%20==0:
            for theset in ['Tr','Va']:
                np.save(thepath_analysis_config+f"loss_{theset}", globals()[f'loss_{theset}'])
                np.save(thepath_analysis_config+f"err_{theset}", globals()[f'err_{theset}']); np.save(thepath_analysis_config+f"err_sum_{theset}", globals()[f'err_sum_{theset}']); np.save(thepath_analysis_config+f"err_tot_{theset}", globals()[f'err_tot_{theset}'])
                np.save(thepath_analysis_config+f"err_cut1_{theset}", globals()[f'err_cut1_{theset}']); np.save(thepath_analysis_config+f"err_cut2_{theset}", globals()[f'err_cut2_{theset}'])
                np.save(thepath_analysis_config+f"acc_{theset}", globals()[f'acc_{theset}'])
                
                for type_i in range(3):
                    np.save(thepath_analysis_config+f"err_{theset}_{type_i}", globals()[f'err_{theset}_{type_i}']); np.save(thepath_analysis_config+f"err_sum_{theset}_{type_i}", globals()[f'err_sum_{theset}_{type_i}'])
                    np.save(thepath_analysis_config+f"err_cut1_{theset}_{type_i}", globals()[f'err_cut1_{theset}_{type_i}']); np.save(thepath_analysis_config+f"err_cut2_{theset}_{type_i}", globals()[f'err_cut2_{theset}_{type_i}'])
                    np.save(thepath_analysis_config+f"acc_{theset}_{type_i}", globals()[f'acc_{theset}_{type_i}'])
                    
                    for c_i in range(num_classes):
                        np.save(thepath_analysis_config+f"precis_{c_i}_{theset}_{type_i}", globals()[f'precis_{c_i}_{theset}_{type_i}'])
                        np.save(thepath_analysis_config+f"recall_{c_i}_{theset}_{type_i}", globals()[f'recall_{c_i}_{theset}_{type_i}'])
            
                for c_i in range(num_classes):
                    np.save(thepath_analysis_config+f"precis_{c_i}_{theset}", globals()[f'precis_{c_i}_{theset}'])
                    np.save(thepath_analysis_config+f"recall_{c_i}_{theset}", globals()[f'recall_{c_i}_{theset}'])
            
            torch.save(model.module.state_dict(), save_path+f"{epoch}.pt")

if dist_rank == 0:
    for theset in ['Tr','Va']:
        np.save(thepath_analysis_config+f"loss_{theset}", globals()[f'loss_{theset}'])
        np.save(thepath_analysis_config+f"err_{theset}", globals()[f'err_{theset}']); np.save(thepath_analysis_config+f"err_sum_{theset}", globals()[f'err_sum_{theset}']); np.save(thepath_analysis_config+f"err_tot_{theset}", globals()[f'err_tot_{theset}'])
        np.save(thepath_analysis_config+f"err_cut1_{theset}", globals()[f'err_cut1_{theset}']); np.save(thepath_analysis_config+f"err_cut2_{theset}", globals()[f'err_cut2_{theset}'])
        np.save(thepath_analysis_config+f"acc_{theset}", globals()[f'acc_{theset}'])
        
        for type_i in range(3):
            np.save(thepath_analysis_config+f"err_{theset}_{type_i}", globals()[f'err_{theset}_{type_i}']); np.save(thepath_analysis_config+f"err_sum_{theset}_{type_i}", globals()[f'err_sum_{theset}_{type_i}'])
            np.save(thepath_analysis_config+f"err_cut1_{theset}_{type_i}", globals()[f'err_cut1_{theset}_{type_i}']); np.save(thepath_analysis_config+f"err_cut2_{theset}_{type_i}", globals()[f'err_cut2_{theset}_{type_i}'])
            np.save(thepath_analysis_config+f"acc_{theset}_{type_i}", globals()[f'acc_{theset}_{type_i}'])
            
            for c_i in range(num_classes):
                np.save(thepath_analysis_config+f"precis_{c_i}_{theset}_{type_i}", globals()[f'precis_{c_i}_{theset}_{type_i}'])
                np.save(thepath_analysis_config+f"recall_{c_i}_{theset}_{type_i}", globals()[f'recall_{c_i}_{theset}_{type_i}'])
    
        for c_i in range(num_classes):
            np.save(thepath_analysis_config+f"precis_{c_i}_{theset}", globals()[f'precis_{c_i}_{theset}'])
            np.save(thepath_analysis_config+f"recall_{c_i}_{theset}", globals()[f'recall_{c_i}_{theset}'])

del model
dist.destroy_process_group()
metadata.close()