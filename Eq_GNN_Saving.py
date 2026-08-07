#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import math
import torch
import torch.nn.functional as F
from fractions import Fraction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arg_main_path",type=str)
args = parser.parse_args()


thepath = args.arg_main_path
thepath_data = thepath+'Data/'
thepath_nuc_data = thepath+'Data/Nuclear_Data/'

exec(open('Various_Functions.py').read())


thenetwork0 = np.loadtxt(thepath_data+'my_net_XRB.dat').astype('int')
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


stable_list = np.load(thepath_nuc_data+'stable_input.npy')

ME_storage = np.load(thepath_nuc_data+'ME_storage.npy')
BE_storage = np.load(thepath_nuc_data+'BE_storage.npy')

Sep_N_storage = np.load(thepath_nuc_data+'Sep_N_storage.npy')
Sep_2N_storage = np.load(thepath_nuc_data+'Sep_2N_storage.npy')
Sep_Z_storage = np.load(thepath_nuc_data+'Sep_Z_storage.npy')
Sep_2Z_storage = np.load(thepath_nuc_data+'Sep_2Z_storage.npy')

bp_storage = np.load(thepath_nuc_data+'bp_storage.npy')
bm_storage = np.load(thepath_nuc_data+'bm_storage.npy')

Sep_A_storage = np.load(thepath_nuc_data+'Sep_A_storage.npy')

        
the_log_subtract_factor = 37
    
the_clip_X_out = 1e-37

the_clip_X_in = 1e-20

the_log_subtract_factor_rate = 26
the_log_rate_max = 16


print('input_del_t_T_rho loading starts')
input_del_t_T_rho = np.load(thepath_data+f'Training_Data/input_del_t_T_rho.npy') # 4: del t, T, Rho, tag for random(0), tracer(1), Random_Tracer(2)
num_data_cut = input_del_t_T_rho.shape[0]

print('map_X loading starts')

in_map_X, out_map_X, out_map_X_err, out_map_X_og_ref, my_network_map = X_data_processing(clip_X_in=the_clip_X_in)

print(input_del_t_T_rho.shape, in_map_X.shape, out_map_X.shape, out_map_X_og_ref.shape)


num_classes = 3

in_1_1_TF = True

flux_TF = True
rates_TF = True

X_Sep_Z_TF = True; X_Sep_2Z_TF = True

    
print('data_normalization_XRB starts')

guess_TF = False

rate_types = ['pg','gp', 'ag','ga', 'pa','ap', 'bp'] # -26, 16

norm_rates_map = np.zeros((num_data_cut, int(1*len(rate_types)), in_map_X.shape[-2], in_map_X.shape[-1]))
rates_sum_map = np.zeros((num_data_cut, 1, in_map_X.shape[-2], in_map_X.shape[-1]))

d = 0
for t in rate_types:
    norm_rates_map[:,d], temp3, temp, temp2 = rate_data_processing_GNN(np.load(thepath_data+f'Training_Data/rate_{t}_map.npy').astype('float64'), t, the_log_subtract_factor_rate, the_log_rate_max)
    rates_sum_map[:,0] += temp
    d+=1


in_data_1, in_data_2, out_data, out_guess, out_data_cl, out_ref_data, out_og_ref_data = data_normalization_XRB(clip_X_in = the_clip_X_in, clip_X_out = the_clip_X_out, log_subtract_factor = the_log_subtract_factor)


N_cut = in_data_2.shape[-2]; Z_cut = in_data_2.shape[-1]

print(N_cut, Z_cut)
print(in_data_1.shape, in_data_2.shape, out_data.shape)
del in_map_X, out_map_X

print('in_data_1 range')
for i in range(in_data_1.shape[1]):
    print(np.min(in_data_1[:,i]), np.max(in_data_1[:,i]))
    
print('in_data_2 range')
for i in range(in_data_2.shape[1]):
    print(np.min(in_data_2[:,i]), np.max(in_data_2[:,i]))

in_data_1_G, in_data_2_G, iso_NZ, in_data_3_G, r_type, i2r_src, i2r_dst, r2i_src, r2i_dst, r2i_coeff, i2i_src, i2i_dst, count_r, count_i, count_ii, out_data_G = get_GNN_data(in_data_1, in_data_2, out_data)
print(in_data_1_G.shape, in_data_2_G.shape, iso_NZ.shape, in_data_3_G.shape, r_type.shape, 
      i2r_src.shape, i2r_dst.shape, r2i_src.shape, r2i_dst.shape, r2i_coeff.shape, i2i_src.shape, i2i_dst.shape, count_r.shape, count_i.shape, count_ii.shape, out_data_G.shape)

cut1_val = 0.0
cut2_val = 7.0

np.savez(
    thepath_data + "Training_Data/metadata.npz",
    N_cut=N_cut,
    Z_cut=Z_cut,
    FiLM_channels_len=in_data_1_G.shape[-1]-1,
    in_channels_iso_len=in_data_2_G.shape[-1],
    in_channels_rea_len=in_data_3_G.shape[-1],
    iso_len=iso_NZ.shape[0],
    reaction_len=r_type.shape[0],
    num_rate_types=len(rate_types),
    net_length=iso_NZ.shape[0],
    num_classes=num_classes,
    clip_X_in=the_clip_X_in,
    log_subtract_factor=the_log_subtract_factor,
    cut1_val=cut1_val,
    cut2_val=cut2_val,
)


print('in_data_2_G range')
for i in range(in_data_2_G.shape[-1]):
    print(np.min(in_data_2_G[:,:,i]), np.max(in_data_2_G[:,:,i]))

print('in_data_3_G range')
for i in range(in_data_3_G.shape[-1]):
    print(np.min(in_data_3_G[:,:,i]), np.max(in_data_3_G[:,:,i]))

out_data_cl_G = get_1D_array(out_data_cl)
out_ref_data_G = get_1D_array(out_ref_data)
out_og_ref_data_G = get_1D_array(out_og_ref_data)

np.save(thepath_data+f'Training_Data/iso_NZ',iso_NZ)
np.save(thepath_data+f'Training_Data/r_type',r_type)
np.save(thepath_data+f'Training_Data/i2r_src',i2r_src)
np.save(thepath_data+f'Training_Data/i2r_dst',i2r_dst)
np.save(thepath_data+f'Training_Data/r2i_src',r2i_src)
np.save(thepath_data+f'Training_Data/r2i_dst',r2i_dst)
np.save(thepath_data+f'Training_Data/r2i_coeff',r2i_coeff)
np.save(thepath_data+f'Training_Data/i2i_src',i2i_src)
np.save(thepath_data+f'Training_Data/i2i_dst',i2i_dst)

np.save(thepath_data+f'Training_Data/slicing_rea',slicing_rea)
np.save(thepath_data+f'Training_Data/slicing_i2r',slicing_i2r)
np.save(thepath_data+f'Training_Data/slicing_r2i',slicing_r2i)
np.save(thepath_data+f'Training_Data/slicing_i2i',slicing_i2i)


p_index_N = 0; p_index_Z = 1
a_index_N = 2; a_index_Z = 2
n_index_N = 1; n_index_Z = 0
c12_index_N = 6; c12_index_Z = 6

p_index_net = np.where((my_network[:,0]==p_index_N) & (my_network[:,1]==p_index_Z))[0][0]
a_index_net = np.where((my_network[:,0]==a_index_N) & (my_network[:,1]==a_index_Z))[0][0]
n_index_net = np.where((my_network[:,0]==n_index_N) & (my_network[:,1]==n_index_Z))[0][0]
c12_index_net = np.where((my_network[:,0]==c12_index_N) & (my_network[:,1]==c12_index_Z))[0][0]


out_data_w_G = np.ones(out_data_G.shape)

out_data_w_G[in_data_1[:,-1]==0] = 0.5
out_data_w_G[in_data_1[:,-1]!=0] = 1

out_data_w_G[:,p_index_net] = out_data_w_G[:,p_index_net]*10
out_data_w_G[:,a_index_net] = out_data_w_G[:,a_index_net]*10

del in_data_1, in_data_2, out_data, out_guess, out_data_cl, out_ref_data, out_og_ref_data



print('shuffling starts')
np.random.seed(0)
temp_i = np.arange(len(in_data_1_G))
np.random.shuffle(temp_i)

the_Tr_ratio = 0.8; the_Va_ratio = 0.1
out_Tr, out_Va, out_Te = split_data(out_data_G[temp_i], Tr_ratio = the_Tr_ratio, Va_ratio = the_Va_ratio)
del out_data_G
np.save(thepath_data+f'Training_Data/out_Tr',out_Tr.astype('float')); np.save(thepath_data+f'Training_Data/out_Va',out_Va.astype('float')); np.save(thepath_data+f'Training_Data/out_Te',out_Te.astype('float'))
print('out saved')
del out_Tr, out_Va, out_Te

out_w_Tr, out_w_Va, out_w_Te = split_data(out_data_w_G[temp_i], Tr_ratio = the_Tr_ratio, Va_ratio = the_Va_ratio)
del out_data_w_G
np.save(thepath_data+f'Training_Data/out_w_Tr',out_w_Tr.astype('float')); np.save(thepath_data+f'Training_Data/out_w_Va',out_w_Va.astype('float')); np.save(thepath_data+f'Training_Data/out_w_Te',out_w_Te.astype('float'))
print('out_w saved')
del out_w_Tr, out_w_Va, out_w_Te

out_og_ref_Tr, out_og_ref_Va, out_og_ref_Te = split_data(out_og_ref_data_G[temp_i], Tr_ratio = the_Tr_ratio, Va_ratio = the_Va_ratio)
del out_og_ref_data_G
np.save(thepath_data+f'Training_Data/out_og_ref_Tr',out_og_ref_Tr.astype('float')); np.save(thepath_data+f'Training_Data/out_og_ref_Va',out_og_ref_Va.astype('float')); np.save(thepath_data+f'Training_Data/out_og_ref_Te',out_og_ref_Te.astype('float'))
print('out_og_ref saved')
del out_og_ref_Tr, out_og_ref_Va, out_og_ref_Te


in_Tr_1, in_Va_1, in_Te_1 = split_data(in_data_1_G[temp_i], Tr_ratio = the_Tr_ratio, Va_ratio = the_Va_ratio)
del in_data_1_G
print('Saving starts')
np.save(thepath_data+f'Training_Data/in_Tr_1',in_Tr_1.astype('float')); np.save(thepath_data+f'Training_Data/in_Va_1',in_Va_1.astype('float')); np.save(thepath_data+f'Training_Data/in_Te_1',in_Te_1.astype('float'))
print('in_1 saved')

in_Tr_3, in_Va_3, in_Te_3 = split_data(in_data_3_G[temp_i], Tr_ratio = the_Tr_ratio, Va_ratio = the_Va_ratio)
del in_data_3_G
print('Saving starts')
np.save(thepath_data+f'Training_Data/in_Tr_3',in_Tr_3.astype('float')); np.save(thepath_data+f'Training_Data/in_Va_3',in_Va_3.astype('float')); np.save(thepath_data+f'Training_Data/in_Te_3',in_Te_3.astype('float'))
print('in_3 saved')

out_ref_Tr, out_ref_Va, out_ref_Te = split_data(out_ref_data_G[temp_i], Tr_ratio = the_Tr_ratio, Va_ratio = the_Va_ratio)
del out_ref_data_G
np.save(thepath_data+f'Training_Data/out_ref_Tr',out_ref_Tr.astype('float')); np.save(thepath_data+f'Training_Data/out_ref_Va',out_ref_Va.astype('float')); np.save(thepath_data+f'Training_Data/out_ref_Te',out_ref_Te.astype('float'))
print('out_ref saved')


len_Tr_type = np.zeros(3); len_Va_type = np.zeros(3); len_Te_type = np.zeros(3)
len_cut1_cut_Tr_type = np.zeros(3); len_cut1_cut_Va_type = np.zeros(3); len_cut1_cut_Te_type = np.zeros(3)
len_cut2_cut_Tr_type = np.zeros(3); len_cut2_cut_Va_type = np.zeros(3); len_cut2_cut_Te_type = np.zeros(3)
for i in range(3):
    theindices_Tr = np.where(in_Tr_1[:,-1]==i)[0]
    theindices_Va = np.where(in_Va_1[:,-1]==i)[0]
    theindices_Te = np.where(in_Te_1[:,-1]==i)[0]
    len_Tr_type[i] = len(theindices_Tr)
    len_Va_type[i] = len(theindices_Va)
    len_Te_type[i] = len(theindices_Te)
    
    len_cut1_cut_Tr_type[i] = len(np.where(abs(out_ref_Tr[theindices_Tr].reshape(-1))>cut1_val)[0])
    len_cut1_cut_Va_type[i] = len(np.where(abs(out_ref_Va[theindices_Va].reshape(-1))>cut1_val)[0])
    len_cut1_cut_Te_type[i] = len(np.where(abs(out_ref_Te[theindices_Te].reshape(-1))>cut1_val)[0])

    len_cut2_cut_Tr_type[i] = len(np.where(abs(out_ref_Tr[theindices_Tr].reshape(-1))>cut2_val)[0])
    len_cut2_cut_Va_type[i] = len(np.where(abs(out_ref_Va[theindices_Va].reshape(-1))>cut2_val)[0])
    len_cut2_cut_Te_type[i] = len(np.where(abs(out_ref_Te[theindices_Te].reshape(-1))>cut2_val)[0])

len_cut1_cut_Tr = np.array([len(np.where(abs(out_ref_Tr.reshape(-1))>cut1_val)[0])])
len_cut1_cut_Va = np.array([len(np.where(abs(out_ref_Va.reshape(-1))>cut1_val)[0])])
len_cut1_cut_Te = np.array([len(np.where(abs(out_ref_Te.reshape(-1))>cut1_val)[0])])

len_cut2_cut_Tr = np.array([len(np.where(abs(out_ref_Tr.reshape(-1))>cut2_val)[0])])
len_cut2_cut_Va = np.array([len(np.where(abs(out_ref_Va.reshape(-1))>cut2_val)[0])])
len_cut2_cut_Te = np.array([len(np.where(abs(out_ref_Te.reshape(-1))>cut2_val)[0])])

del out_ref_Tr, out_ref_Va, out_ref_Te

np.save(thepath_data+f'Training_Data/len_Tr_type',len_Tr_type); np.save(thepath_data+f'Training_Data/len_Va_type',len_Va_type); np.save(thepath_data+f'Training_Data/len_Te_type',len_Te_type)
np.save(thepath_data+f'Training_Data/len_cut1_cut_Tr_type',len_cut1_cut_Tr_type); np.save(thepath_data+f'Training_Data/len_cut1_cut_Va_type',len_cut1_cut_Va_type); np.save(thepath_data+f'Training_Data/len_cut1_cut_Te_type',len_cut1_cut_Te_type)
np.save(thepath_data+f'Training_Data/len_cut2_cut_Tr_type',len_cut2_cut_Tr_type); np.save(thepath_data+f'Training_Data/len_cut2_cut_Va_type',len_cut2_cut_Va_type); np.save(thepath_data+f'Training_Data/len_cut2_cut_Te_type',len_cut2_cut_Te_type)

np.save(thepath_data+f'Training_Data/len_cut1_cut_Tr',len_cut1_cut_Tr); np.save(thepath_data+f'Training_Data/len_cut1_cut_Va',len_cut1_cut_Va); np.save(thepath_data+f'Training_Data/len_cut1_cut_Te',len_cut1_cut_Te)
np.save(thepath_data+f'Training_Data/len_cut2_cut_Tr',len_cut2_cut_Tr); np.save(thepath_data+f'Training_Data/len_cut2_cut_Va',len_cut2_cut_Va); np.save(thepath_data+f'Training_Data/len_cut2_cut_Te',len_cut2_cut_Te)



print('out_data_cl_G process starts')
out_cl_Tr, out_cl_Va, out_cl_Te = split_data(out_data_cl_G[temp_i], Tr_ratio = the_Tr_ratio, Va_ratio = the_Va_ratio)
del out_data_cl_G

len_Tr_true_classes = np.zeros(num_classes); len_Va_true_classes = np.zeros(num_classes); len_Te_true_classes = np.zeros(num_classes)
for i in range(num_classes):
    len_Tr_true_classes[i] = len(np.where(out_cl_Tr==i)[0])
    len_Va_true_classes[i] = len(np.where(out_cl_Va==i)[0])
    len_Te_true_classes[i] = len(np.where(out_cl_Te==i)[0])

len_Tr_type_true_classes = np.zeros((3,num_classes)); len_Va_type_true_classes = np.zeros((3,num_classes)); len_Te_type_true_classes = np.zeros((3,num_classes))
for i in range(3):
    theindices_Tr = np.where(in_Tr_1[:,-1]==i)[0]
    theindices_Va = np.where(in_Va_1[:,-1]==i)[0]
    theindices_Te = np.where(in_Te_1[:,-1]==i)[0]
    for j in range(num_classes):
        len_Tr_type_true_classes[i,j] = len(np.where(out_cl_Tr[theindices_Tr]==j)[0])
        len_Va_type_true_classes[i,j] = len(np.where(out_cl_Va[theindices_Va]==j)[0])
        len_Te_type_true_classes[i,j] = len(np.where(out_cl_Te[theindices_Te]==j)[0])

np.save(thepath_data+f'Training_Data/len_Tr_true_classes',len_Tr_true_classes)
np.save(thepath_data+f'Training_Data/len_Va_true_classes',len_Va_true_classes)
np.save(thepath_data+f'Training_Data/len_Te_true_classes',len_Te_true_classes)
np.save(thepath_data+f'Training_Data/len_Tr_type_true_classes',len_Tr_type_true_classes)
np.save(thepath_data+f'Training_Data/len_Va_type_true_classes',len_Va_type_true_classes)
np.save(thepath_data+f'Training_Data/len_Te_type_true_classes',len_Te_type_true_classes)

del in_Tr_1, in_Va_1, in_Te_1

np.save(thepath_data+f'Training_Data/out_cl_Tr',out_cl_Tr); np.save(thepath_data+f'Training_Data/out_cl_Va',out_cl_Va); np.save(thepath_data+f'Training_Data/out_cl_Te',out_cl_Te)

del out_cl_Tr, out_cl_Va, out_cl_Te


in_Tr_2, in_Va_2, in_Te_2 = split_data(in_data_2_G[temp_i], Tr_ratio = the_Tr_ratio, Va_ratio = the_Va_ratio)
del in_data_2_G

np.save(thepath_data+f'Training_Data/in_Va_2',in_Va_2.astype('float'))
del in_Va_2
print('Va saved')

np.save(thepath_data+f'Training_Data/in_Te_2',in_Te_2.astype('float'))
del in_Te_2
print('Te saved')

print('Saving starts')
np.save(thepath_data+f'Training_Data/in_Tr_2',in_Tr_2.astype('float'))
del in_Tr_2
print('Tr saved')




