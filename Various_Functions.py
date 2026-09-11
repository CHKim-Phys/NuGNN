#!/usr/bin/env python
# coding: utf-8

# In[5]:


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# In[6]:


def network_map_converter(subtract_factor=0): # (num_data, net_length, 3); 3: N, Z, X
    
    temp_N_max = int(np.max(my_network[:,0]-subtract_factor*my_network[:,1]))
    temp_N_min = int(np.min(my_network[:,0]-subtract_factor*my_network[:,1]))
    N_len = temp_N_max - temp_N_min + 1
    Z_len = themaxZ
    
    themap = np.zeros((N_len, Z_len))
    for i in range(len(my_network)):
        theN = int(my_network[i,0])
        theZ = int(my_network[i,1])
        themap[theN-int(subtract_factor*theZ)-temp_N_min,theZ] = 1

    return themap


# In[ ]:


def two_diff(theA, theB):
    # Error-free transform (Dekker/Knuth)
    theval = theA - theB
    theres  = theval - theA
    theerr = (theA - (theval - theres)) - (theB + theres)
    return theval, theerr

def two_sum(theA, theB):
    # Error-free transform (Dekker/Knuth)
    theval  = theA + theB
    theres = theval - theA
    theerr = (theA - (theval - theres)) + (theB - theres)
    return theval, theerr


def X_data_processing(clip_X_in = 1e-30):
    print('map_X loading starts')
    the_in_X = np.load(thepath_data+f'Training_Data/in_map_X.npy').astype(np.float128)#[type_indices]
    the_out_X_og = np.load(thepath_data+f'Training_Data/out_map_X.npy').astype(np.float128)#[type_indices]
    print(the_in_X.shape)
    print(the_out_X_og.shape)
    
    the_net_map = globals()[f'my_network_map'].copy()
            
    del_val, del_err = two_diff(the_out_X_og, the_in_X)
    
    thedelta1, thedelta2 = two_sum(del_val, del_err)
    the_out_X = thedelta1+thedelta2
    the_out_X_err = del_err
        
    theindices = np.where((the_in_X==0) & (the_out_X_og<clip_X_in))
        
    the_out_X[theindices] = 0

    return the_in_X, the_out_X, the_out_X_err, the_out_X_og, the_net_map
    

def data_normalization_XRB(clip_X_in = 1e-30, clip_X_out = 1e-47, log_subtract_factor=47):
    
    d_num = 4    
    norm_in_del_t_T_rho = np.zeros((input_del_t_T_rho.shape[0], d_num))
    norm_in_del_t_T_rho[:,:3] = np.log10(input_del_t_T_rho[:,:3].copy())

    if in_1_1_TF:
        norm_in_del_t_T_rho[:,0] = norm_in_del_t_T_rho[:,0].copy()/6+1.6 # log10(del_t)/8+1.3
        norm_in_del_t_T_rho[:,1] = norm_in_del_t_T_rho[:,1].copy()/1.8+0.7 # log10(T9)/2+0.5
        norm_in_del_t_T_rho[:,2] = norm_in_del_t_T_rho[:,2].copy()/5.5-0.6 # log10(rho)/10-0.45
    else:
        norm_in_del_t_T_rho[:,0] = norm_in_del_t_T_rho[:,0].copy()/20+0.5 # log10(del_t)/20+0.5
        norm_in_del_t_T_rho[:,1] = norm_in_del_t_T_rho[:,1].copy()/4+0.2 # log10(T9)/4+0.2
        norm_in_del_t_T_rho[:,2] = norm_in_del_t_T_rho[:,2].copy()/15-0.3 # log10(rho)/15-0.3      

    d_temp = 3
        
    p_index_N = 0; p_index_Z = 1
    a_index_N = 2; a_index_Z = 2
    n_index_N = 1; n_index_Z = 0

        
    norm_in_del_t_T_rho[:,d_temp] = input_del_t_T_rho[:,-1].copy(); d_temp+=1

    d_num = 1
    if flux_TF: d_num += 1 # for flux
    if rates_TF: d_num += norm_rates_map.shape[1]
    
    if X_Sep_Z_TF: d_num+=1
    if X_Sep_2Z_TF: d_num+=1
        
    norm_in_map_X = np.zeros((in_map_X.shape[0],d_num,in_map_X.shape[1],in_map_X.shape[2]))
    norm_in_map_X[:,0] = in_map_X.copy()
    norm_in_map_X[:,0][np.where(in_map_X<=clip_X_in)] = clip_X_in # set 0 -> -30
    norm_in_map_X[:,0] = np.log10(norm_in_map_X[:,0].copy())/abs(np.log10(clip_X_in)/2)+1 # log10(X)/15+1
    
    d_temp = 1
    
    theflux = np.load(thepath_data+f'Training_Data/flux_map.npy')[:num_data_cut]*input_del_t_T_rho[:,0].reshape(-1,1,1)#[type_indices]
    if flux_TF:
        globals()['the_log_flux_max'] = np.log10(np.max(abs(theflux[theflux!=0])))+1
        globals()['the_log_flux_min'] = np.min(np.log10(abs(theflux[theflux!=0])))-1
        norm_in_map_X[:,d_temp] = np.sign(theflux)*(-the_log_flux_min+np.log10(np.clip(abs(theflux),10**(the_log_flux_min),10**(the_log_flux_max))))/(the_log_flux_max-the_log_flux_min)
        d_temp +=1

    if rates_TF:
        norm_in_map_X[:,d_temp:d_temp+norm_rates_map.shape[1]] = norm_rates_map
        d_temp += int(norm_rates_map.shape[1])
        
    temp_N_max = int(np.max(my_network[:,0]))
    temp_N_min = int(np.min(my_network[:,0]))
    
    X_mask_i = np.where(norm_in_map_X[:,0]==-1)

    if X_Sep_Z_TF:
        X_Sep_Z_temp = np.zeros(in_map_X.shape)
        for n_z in range(in_map_X.shape[-2]):
            for z in range(in_map_X.shape[-1]):
                N_temp = (n_z+temp_N_min)
                Z_temp = z
                if len(np.where((Sep_Z_storage[:,0]==N_temp) & (Sep_Z_storage[:,1]==Z_temp))[0])==1:
                    thei = np.where((Sep_Z_storage[:,0]==N_temp) & (Sep_Z_storage[:,1]==Z_temp))[0]
                    X_Sep_Z_temp[:,n_z,z] = Sep_Z_storage[thei,2]/input_del_t_T_rho[:,1]/33000/1e2
                else:
                    X_Sep_Z_temp[:,n_z,z] = -1
        norm_in_map_X[:,d_temp] = X_Sep_Z_temp; norm_in_map_X[:,d_temp][X_mask_i] = -1; d_temp+=1; del X_Sep_Z_temp

    if X_Sep_2Z_TF:
        X_Sep_2Z_temp = np.zeros(in_map_X.shape)
        for n_z in range(in_map_X.shape[-2]):
            for z in range(in_map_X.shape[-1]):
                N_temp = (n_z+temp_N_min)
                Z_temp = z
                if len(np.where((Sep_2Z_storage[:,0]==N_temp) & (Sep_2Z_storage[:,1]==Z_temp))[0])==1:
                    thei = np.where((Sep_2Z_storage[:,0]==N_temp) & (Sep_2Z_storage[:,1]==Z_temp))[0]
                    X_Sep_2Z_temp[:,n_z,z] = Sep_2Z_storage[thei,2]/input_del_t_T_rho[:,1]/60000/1e2
                else:
                    X_Sep_2Z_temp[:,n_z,z] = -1
        norm_in_map_X[:,d_temp] = X_Sep_2Z_temp; norm_in_map_X[:,d_temp][X_mask_i] = -1; d_temp+=1; del X_Sep_2Z_temp

    theguess = theflux#*input_del_t_T_rho[:,0].reshape(-1,1,1)

    theguess = np.sign(theflux)*(log_subtract_factor+np.log10(np.clip(abs(theguess),10**(-log_subtract_factor),1)))/log_subtract_factor
        
    temp_in = in_map_X.copy()
    temp_in[temp_in==0] = 1e-20
    rel_del_X = out_map_X/temp_in
    log_out_map_X = np.sign(rel_del_X.copy())*(np.log10(np.clip(abs(rel_del_X.copy()),1e-17,1e20))+17)/37

    norm_out_map_X = log_out_map_X.copy()
    norm_out_map_X[abs(rel_del_X)<1e-15] = 0
    log_out_map_X[abs(rel_del_X)<1e-15] = 0
    
    
    log_out_map_X = np.sign(rel_del_X.copy())*(np.log10(np.clip(abs(rel_del_X.copy()),1e-17,1e20))+17)
    log_out_map_X[abs(rel_del_X)<1e-15] = 0

    theval, theerr = two_sum(out_map_X.copy(), in_map_X.copy())
    out_map_X_og = theval+(theerr+out_map_X_err)
    out_map_X_og[np.where(out_map_X_og<=clip_X_in)] = clip_X_in # clipping
    out_map_X_og = np.log10(out_map_X_og.copy())

    out_classes = np.zeros(out_map_X.shape)
    
    out_classes[out_map_X==0] = 0
    out_classes[out_map_X>0] = 1
    out_classes[out_map_X<0] = 2
        
    return norm_in_del_t_T_rho, norm_in_map_X, norm_out_map_X, theguess, out_classes, log_out_map_X, out_map_X_og


def rate_data_processing_GNN(therates, theid, log_subtract_factor, log_rate_max):

    therates_prod = np.zeros(therates.shape)
    p_index_N = 0; p_index_Z = 1
    a_index_N = 2; a_index_Z = 2
    n_index_N = 1; n_index_Z = 0

    if theid=='pg': therates_prod[:,:,1:] = therates[:,:,:-1]
    elif theid=='gp': therates_prod[:,:,:-1] = therates[:,:,1:]
        
    elif theid=='ag': therates_prod[:,2:,2:] = therates[:,:-2,:-2]
    elif theid=='ga': therates_prod[:,:-2,:-2] = therates[:,2:,2:]

    elif theid=='ng': therates_prod[:,1:,:] = therates[:,:-1,:]
    elif theid=='gn': therates_prod[:,:-1,:] = therates[:,1:,:]    
        
    elif theid=='ap': therates_prod[:,2:,1:] = therates[:,:-2,:-1]
    elif theid=='pa': therates_prod[:,:-2,:-1] = therates[:,2:,1:]
        
    elif theid=='pn': therates_prod[:,1:,:-1] = therates[:,:-1,1:]
    elif theid=='np': therates_prod[:,:-1,1:] = therates[:,1:,:-1]
        
    elif theid=='an': therates_prod[:,1:,2:] = therates[:,:-1,:-2]
    elif theid=='na': therates_prod[:,:-1,:-2] = therates[:,1:,2:]
        
    elif theid=='bp': therates_prod[:,1:,:-1] = therates[:,:-1,1:]
    elif theid=='bm': therates_prod[:,:-1,1:] = therates[:,1:,:-1]

    norm_therates = (log_subtract_factor+np.log10(np.clip(abs(therates), 10**(-log_subtract_factor), 10**(log_rate_max))))/(log_rate_max+log_subtract_factor)
    norm_therates_prod = (log_subtract_factor+np.log10(np.clip(abs(therates_prod), 10**(-log_subtract_factor), 10**(log_rate_max))))/(log_rate_max+log_subtract_factor)
        
    return norm_therates, norm_therates_prod, -therates, therates_prod

def get_1D_array(thedata_map):
        
    thearray = np.zeros((len(thedata_map), net_length))
    for i in range(len(my_network)):
        N = int(my_network[i,0]); Z = int(my_network[i,1])
        thearray[:,i] = thedata_map[:,N,Z]

    return thearray

def get_product_indices(N_change,Z_change,t):

    theindices_r = np.zeros(net_length)
    theindices_i = np.zeros(net_length)

    d = 0
    for i in range(net_length):
        N = int(my_network[i,0]); Z = int(my_network[i,1])
        if t=='pg' and N==0 and Z==1:
            N_new = N+1; Z_new = Z+0
        elif t=='ag' and N==2 and Z==2:
            N_new = N+4; Z_new = Z+4 # c12
        elif t=='ga' and N==6 and Z==6:
            N_new = 2; Z_new = 2     # c12
        else:
            N_new = N+N_change; Z_new = Z+Z_change
            
        thei = np.where((my_network[:,0]==N_new) & (my_network[:,1]==Z_new))[0]
        if len(thei)==1:
            theindices_r[d] = i
            theindices_i[d] = thei[0]
            d+=1

    return theindices_r[:d].astype('int'), theindices_i[:d].astype('int')


def get_GNN_data(the_norm_in_del_t_T_rho, the_norm_in_map_X, the_norm_out_map_X):
    
    p_index_N = 0; p_index_Z = 1
    a_index_N = 2; a_index_Z = 2
    n_index_N = 1; n_index_Z = 0
    c12_index_N = 6; c12_index_Z = 6

    p_index_net = np.where((my_network[:,0]==p_index_N) & (my_network[:,1]==p_index_Z))[0][0]
    a_index_net = np.where((my_network[:,0]==a_index_N) & (my_network[:,1]==a_index_Z))[0][0]
    n_index_net = np.where((my_network[:,0]==n_index_N) & (my_network[:,1]==n_index_Z))[0][0]
    c12_index_net = np.where((my_network[:,0]==c12_index_N) & (my_network[:,1]==c12_index_Z))[0][0]

    # add edge info too? by source=0, others=1 ###
    input_iso = np.zeros((the_norm_in_map_X.shape[0],net_length,the_norm_in_map_X.shape[1]+the_norm_in_del_t_T_rho.shape[-1]-1))
    # NZ info for embedding
    the_iso_NZ = np.zeros((net_length,2))
    
    d = 0
    for i in range(the_norm_in_map_X.shape[1]):
        input_iso[:,:,d] = get_1D_array(the_norm_in_map_X[:,d]); d+=1
    input_iso[:,:,d:] = the_norm_in_del_t_T_rho[:,None,:-1]
    
    the_iso_NZ[:,:2] = my_network.copy().astype('int')
    
    the_iso_NZ = the_iso_NZ.astype('int')

    
    globals()['reaction_features_len'] = 1+the_norm_in_del_t_T_rho.shape[-1]-1
    
    input_rea = np.zeros((the_norm_in_map_X.shape[0],int(net_length*len(rate_types)),globals()['reaction_features_len']))
    the_r_type = np.zeros((int(net_length*len(rate_types))))
    input_rea[:,:,1:] = the_norm_in_del_t_T_rho[:,None,:-1]

    the_i2r_src = np.zeros((int(net_length*len(rate_types)*4)))
    the_i2r_dst = np.zeros((int(net_length*len(rate_types)*4)))

    the_r2i_src = np.zeros((int(net_length*len(rate_types)*5)))
    the_r2i_dst = np.zeros((int(net_length*len(rate_types)*5)))
    the_r2i_coeff = np.zeros((int(net_length*len(rate_types)*5)))

    globals()['slicing_rea'] = [0]
    globals()['slicing_i2r'] = [0]
    globals()['slicing_r2i'] = [0]
    
    globals()['r_indices_list'] = []
    
    d = 0
    d_r_i = 0
    d_i_i = 0
    for i in range(len(rate_types)):
        t = rate_types[i]
        temp, temp0, temp1, temp2 = rate_data_processing_GNN(np.load(thepath_data+f'Training_Data/rate_{t}_map.npy').astype('float64')[:num_data_cut], t, the_log_subtract_factor_rate, the_log_rate_max)
        
        if len(t)==2:
            N_change = 0; Z_change = 0
            if t=='bp': N_change=1; Z_change=-1
            elif t=='bm': N_change=-1; Z_change=1
            elif t[0]=='p': Z_change+=1
            elif t[0]=='a': N_change+=2; Z_change+=2
            elif t[0]=='n': N_change+=1

            if t=='bp': N_change=1; Z_change=-1
            elif t=='bm': N_change=-1; Z_change=1
            elif t[1]=='p': Z_change-=1
            elif t[1]=='a': N_change-=2; Z_change-=2
            elif t[1]=='n': N_change-=1

        else:
            print(t, 'not ready')

        indices_r, indices_i = get_product_indices(N_change,Z_change,t)
        
        globals()['r_indices_list'].append(indices_r)

        len_ri = len(indices_r)
        
        # input_r (reaction_len,1), 1 for rate value
        input_rea[:,d:d+len_ri,0] = get_1D_array(temp)[:,indices_r]
        # r_type (reaction_len), rate type for embedding
        the_r_type[d:d+len_ri] = i

        # indices for isotopes (input_i) to reactions (input_r); heavy target
        the_i2r_src[d_r_i:d_r_i+len_ri] = indices_r
        the_i2r_dst[d_r_i:d_r_i+len_ri] = np.arange(d,d+len_ri)
        d_r_i+=len_ri

        # indices for reactions (input_r) to isotopes (input_i); heavy target
        the_r2i_src[d_i_i:d_i_i+len_ri] = np.arange(d,d+len_ri)
        the_r2i_dst[d_i_i:d_i_i+len_ri] = indices_r
        the_r2i_coeff[d_i_i:d_i_i+len_ri] = -1
        d_i_i+=len_ri

        # indices for isotopes (input_i) to reactions (input_r); light incident
        # indices for reactions (input_r) to isotopes (input_i); light incident
        if t=='pg' or t=='pa' or t=='pn':
            the_i2r_src[d_r_i:d_r_i+len_ri] = np.ones(len_ri)*p_index_net
            the_i2r_dst[d_r_i:d_r_i+len_ri] = np.arange(d,d+len_ri)
            d_r_i+=len_ri
            
            the_r2i_src[d_i_i:d_i_i+len_ri] = np.arange(d,d+len_ri)
            the_r2i_dst[d_i_i:d_i_i+len_ri] = np.ones(len_ri)*p_index_net
            the_r2i_coeff[d_i_i:d_i_i+len_ri] = -1
            d_i_i+=len_ri
            
        elif t=='ag' or t=='ap' or t=='an':
            the_i2r_src[d_r_i:d_r_i+len_ri] = np.ones(len_ri)*a_index_net
            the_i2r_dst[d_r_i:d_r_i+len_ri] = np.arange(d,d+len_ri)
            d_r_i+=len_ri

            the_r2i_src[d_i_i:d_i_i+len_ri] = np.arange(d,d+len_ri)
            the_r2i_dst[d_i_i:d_i_i+len_ri] = np.ones(len_ri)*a_index_net
            the_r2i_coeff[d_i_i:d_i_i+len_ri] = -1
            d_i_i+=len_ri

            ### triple alpha ###
            if t=='ag':
                temp_i = np.where(indices_r==a_index_net)[0][0]
                
                the_i2r_src[d_r_i:d_r_i+1] = np.ones(1)*a_index_net
                the_i2r_dst[d_r_i:d_r_i+1] = d+temp_i
                d_r_i+=1
                
                the_r2i_src[d_i_i:d_i_i+1] = d+temp_i
                the_r2i_dst[d_i_i:d_i_i+1] = np.ones(1)*a_index_net
                the_r2i_coeff[d_i_i:d_i_i+1] = -1
                d_i_i+=1
        
        elif t=='ng' or t=='np' or t=='na':
            the_i2r_src[d_r_i:d_r_i+len_ri] = np.ones(len_ri)*n_index_net
            the_i2r_dst[d_r_i:d_r_i+len_ri] = np.arange(d,d+len_ri)
            d_r_i+=len_ri

            the_r2i_src[d_i_i:d_i_i+len_ri] = np.arange(d,d+len_ri)
            the_r2i_dst[d_i_i:d_i_i+len_ri] = np.ones(len_ri)*n_index_net
            the_r2i_coeff[d_i_i:d_i_i+len_ri] = -1
            d_i_i+=len_ri        

        # indices for reactions (input_r) to isotopes (input_i); light ejectile
        if t=='gp' or t=='ap' or t=='np':
            the_r2i_src[d_i_i:d_i_i+len_ri] = np.arange(d,d+len_ri)
            the_r2i_dst[d_i_i:d_i_i+len_ri] = np.ones(len_ri)*p_index_net
            the_r2i_coeff[d_i_i:d_i_i+len_ri] = 1
            d_i_i+=len_ri
            
        elif t=='ga' or t=='pa' or t=='na':
            the_r2i_src[d_i_i:d_i_i+len_ri] = np.arange(d,d+len_ri)
            the_r2i_dst[d_i_i:d_i_i+len_ri] = np.ones(len_ri)*a_index_net
            the_r2i_coeff[d_i_i:d_i_i+len_ri] = 1
            d_i_i+=len_ri

            ### triple alpha ###
            if t=='ga':
                temp_i = np.where(indices_r==c12_index_net)[0][0]
                
                the_r2i_src[d_i_i:d_i_i+1] = d+temp_i
                the_r2i_dst[d_i_i:d_i_i+1] = np.ones(1)*a_index_net
                the_r2i_coeff[d_i_i:d_i_i+1] = 1
                d_i_i+=1
            
        elif t=='gn' or t=='pn' or t=='an':
            the_r2i_src[d_i_i:d_i_i+len_ri] = np.arange(d,d+len_ri)
            the_r2i_dst[d_i_i:d_i_i+len_ri] = np.ones(len_ri)*n_index_net
            the_r2i_coeff[d_i_i:d_i_i+len_ri] = 1
            d_i_i+=len_ri

        # indices for reactions (input_r) to isotopes (input_i); heavy ejectile
        the_r2i_src[d_i_i:d_i_i+len_ri] = np.arange(d,d+len_ri)
        the_r2i_dst[d_i_i:d_i_i+len_ri] = indices_i
        the_r2i_coeff[d_i_i:d_i_i+len_ri] = 1
        d_i_i+=len_ri
            
        d+=len_ri

        globals()['slicing_rea'].append(d)
        globals()['slicing_i2r'].append(d_r_i)
        globals()['slicing_r2i'].append(d_i_i)
    

    globals()['iso_len'] = net_length
    globals()['reaction_len'] = d
    globals()['num_rate_types'] = len(rate_types)
    
    input_rea = input_rea[:,:d]
    the_r_type = the_r_type[:d].astype('int')

    the_i2r_src = the_i2r_src[:d_r_i].astype('int')
    the_i2r_dst = the_i2r_dst[:d_r_i].astype('int')

    the_r2i_src = the_r2i_src[:d_i_i].astype('int')
    the_r2i_dst = the_r2i_dst[:d_i_i].astype('int')
    the_r2i_coeff = the_r2i_coeff[:d_i_i]

    the_count_r = np.bincount(the_i2r_dst, minlength=reaction_len)   # (reaction_len)
    the_count_i = np.bincount(the_r2i_dst, minlength=iso_len)        # (iso_len)
    
    the_count_r = the_count_r + 1
    the_count_i = the_count_i + 1

    
    the_i2i_src = np.zeros((int(net_length*len(rate_types)*25)))
    the_i2i_dst = np.zeros((int(net_length*len(rate_types)*25)))
    globals()['slicing_i2i'] = [0]

    d = 0
    for t in range(num_rate_types):
        rea_s, rea_e = slicing_rea[t], slicing_rea[t+1]
    
        # build a lookup reaction -> list of isotopes
        for r in range(rea_s, rea_e):
            nodes = np.unique(the_r2i_dst[the_r2i_src == r])
            d_a = 0
            for node_a in nodes:
                d_b = 0
                for node_b in nodes:
                    if node_a == node_b:
                        continue
                    
                    the_i2i_src[d] = node_a
                    the_i2i_dst[d] = node_b
                    d+=1
                    d_b+=1
                d_a+=1
    
        globals()['slicing_i2i'].append(d)

    the_i2i_src = the_i2i_src[:d].astype('int')
    the_i2i_dst = the_i2i_dst[:d].astype('int')

    the_count_ii = np.bincount(the_i2i_dst, minlength=iso_len)        # (iso_len)
    the_count_ii = the_count_ii + 1
    
    the_norm_out_map_X = get_1D_array(the_norm_out_map_X)
    
    return the_norm_in_del_t_T_rho, input_iso, the_iso_NZ, input_rea, the_r_type, the_i2r_src, the_i2r_dst, the_r2i_src, the_r2i_dst, the_r2i_coeff, the_i2i_src, the_i2i_dst, the_count_r, the_count_i, the_count_ii, the_norm_out_map_X




def split_data(numpy_data, Tr_ratio = 0.7, Va_ratio = 0.15):
    train_size = int(round(numpy_data.shape[0]*Tr_ratio))
    val_size = int(round(numpy_data.shape[0]*(Tr_ratio+Va_ratio)))

    train_data = numpy_data[:train_size]
    val_data = numpy_data[train_size:val_size]
    test_data = numpy_data[val_size:]
    
    return train_data, val_data, test_data


def Error_RG(pred, targets):
    
    abs_mean_error = np.sum(abs(targets-pred))
    abs_rel_mean_error = np.sum(abs((targets-pred)/(targets+1e-30)))
    
    return abs_mean_error, abs_rel_mean_error


class NumpyMemMap_G(torch.utils.data.Dataset):
    def __init__(self, path_in_1, path_in_2, path_in_3, path_out, path_out_w, path_out_cl, path_out_ref, path_out_og_ref, dtype=np.float32):
        self.path_in_1, self.dtype = path_in_1, dtype
        self.path_in_2 = path_in_2
        self.path_in_3 = path_in_3
        self.path_out = path_out
        self.path_out_w = path_out_w
        self.path_out_cl = path_out_cl
        self.path_out_ref = path_out_ref
        self.path_out_og_ref = path_out_og_ref
        
        self.x_in_1 = None
        self.x_in_2 = None
        self.x_in_3 = None
        self.x_out = None
        self.x_out_w = None
        self.x_out_cl = None
        self.x_out_ref = None
        self.x_out_og_ref = None
        
        t = np.load(self.path_in_1, mmap_mode="r"); self.N = len(t); del t
        
    def __len__(self): return self.N

    def __getitem__(self, i):
        if self.x_in_1 is None:                       # opened in THIS worker/rank
            self.x_in_1 = np.load(self.path_in_1, mmap_mode="r")
        if self.x_in_2 is None:                       # opened in THIS worker/rank
            self.x_in_2 = np.load(self.path_in_2, mmap_mode="r")
        if self.x_in_3 is None:                       # opened in THIS worker/rank
            self.x_in_3 = np.load(self.path_in_3, mmap_mode="r")
        if self.x_out is None:                       # opened in THIS worker/rank
            self.x_out = np.load(self.path_out, mmap_mode="r")
        if self.x_out_w is None:                       # opened in THIS worker/rank
            self.x_out_w = np.load(self.path_out_w, mmap_mode="r")
        if self.x_out_cl is None:                       # opened in THIS worker/rank
            self.x_out_cl = np.load(self.path_out_cl, mmap_mode="r")
        if self.x_out_ref is None:                       # opened in THIS worker/rank
            self.x_out_ref = np.load(self.path_out_ref, mmap_mode="r")
        if self.x_out_og_ref is None:                       # opened in THIS worker/rank
            self.x_out_og_ref = np.load(self.path_out_og_ref, mmap_mode="r")
            
        in_1 = torch.tensor(np.asarray(self.x_in_1[i], dtype=self.dtype, order="C"))
        in_2 = torch.tensor(np.asarray(self.x_in_2[i], dtype=self.dtype, order="C"))
        in_3 = torch.tensor(np.asarray(self.x_in_3[i], dtype=self.dtype, order="C"))
        out = torch.tensor(np.asarray(self.x_out[i], dtype=self.dtype, order="C"))
        out_w = torch.tensor(np.asarray(self.x_out_w[i], dtype=self.dtype, order="C"))
        out_cl = torch.tensor(np.asarray(self.x_out_cl[i], order="C"))
        out_ref = torch.tensor(np.asarray(self.x_out_ref[i], order="C"))
        out_og_ref = torch.tensor(np.asarray(self.x_out_og_ref[i], order="C"))
            
        return in_1, in_2, in_3, out, out_w, out_cl, out_ref, out_og_ref

