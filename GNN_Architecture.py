class FiLM_layer(nn.Module):
    def __init__(self, thesize=0):
        super().__init__()
        
        layers = []
        for i in range(param_dic_FiLM['num_lin']):
            if i==0:
                layers.append(nn.Linear(FiLM_channels_len,param_dic_FiLM[f'lin{i+1}']))
            elif i!=param_dic_FiLM['num_lin']-1:
                layers.append(nn.Linear(param_dic_FiLM[f'lin{i}'],param_dic_FiLM[f'lin{i+1}']))
            else:
                if thesize==0:
                    layers.append(nn.Linear(param_dic_FiLM[f'lin{i}'],param_dic_FiLM[f'lin{i+1}']))
                else:
                    layers.append(nn.Linear(param_dic_FiLM[f'lin{i}'],int(thesize)))
                
        self.layers = nn.ModuleList(layers)
        
        self.act = nn.LeakyReLU(inplace=False)

        
    def forward(self, inputs):
            
        out = self.act(self.layers[0](inputs))
        for layer in self.layers[1:-1]:
            out = self.act(layer(out))

        out = self.layers[-1](out)
        
        return out



def scatter_softmax(score: torch.Tensor, dst: torch.Tensor, num_dst: int, eps: float = 1e-12):
    """
    score: [B, E] or [B, E, H]
    dst:   [E] (long)
    returns alpha with same shape as score, softmax separately per destination node.
    """

    B, E, H = score.shape
    dst = dst.view(1, E, 1).expand(B, E, H)  # [B,E,H]

    # max per destination (numerical stability)
    max_per_dst = torch.full((B, num_dst, H), -torch.inf, device=score.device, dtype=score.dtype)
    max_per_dst.scatter_reduce_(1, dst, score, reduce="amax", include_self=True)

    score = score - max_per_dst.gather(1, dst)
    exp = torch.exp(score)

    denom = torch.zeros((B, num_dst, H), device=score.device, dtype=score.dtype)
    denom.scatter_add_(1, dst, exp)

    alpha = exp / (denom.gather(1, dst) + eps)
    
    return alpha

class MPNNLayer_GAT(nn.Module):
    def __init__(self, in_i_chan, in_r_chan, out_i_chan, out_r_chan, NZ_emb_TF = False, r_base_TF = False, num_heads=8):
        super().__init__()

        self.act = nn.LeakyReLU(inplace=False)

        droprate = 0.1
        
        self.drop2d_msg = nn.Dropout2d(droprate)
        

        self.num_heads = num_heads
        
        assert out_i_chan % self.num_heads == 0
        assert out_r_chan % self.num_heads == 0

        self.d_i = out_i_chan // self.num_heads
        self.d_r = out_r_chan // self.num_heads
        
        self.iso_len = iso_len
        self.reaction_len = reaction_len
        
        self.out_i_chan = out_i_chan
        self.out_r_chan = out_r_chan
        
        self.NZ_emb_TF = NZ_emb_TF
        self.r_base_TF = r_base_TF

        if NZ_emb_TF:
            self.emb_len = 1
            self.emb_N = nn.Embedding(N_cut, self.emb_len)
            self.emb_Z = nn.Embedding(Z_cut, self.emb_len)

            emb_in_channels_len = int(self.emb_len*iso_NZ.shape[-1])

            self.in_i_chan = in_i_chan + emb_in_channels_len
        else:
            emb_in_channels_len = 0
             
            self.in_i_chan = in_i_chan + emb_in_channels_len
            
        
        if r_base_TF:
            emb_len = 1
            self.emb_types = nn.Embedding(num_rate_types, emb_len)

            # Build reaction "base" features from reaction attributes (type, log_rate, Q, ...)
            self.r_base = nn.ModuleList()
            for i in range(num_rate_types):
                self.r_base.append(nn.Sequential(
                    nn.Linear(emb_len + in_r_chan, int(2*self.out_r_chan)),
                    self.act,
                    nn.Linear(int(2*self.out_r_chan), self.out_r_chan),
                ))
            self.in_r_chan = self.out_r_chan
          #  self.in_r_chan = emb_len + in_r_chan
        else:
            self.in_r_chan = in_r_chan
        
        # Messages from iso -> iso
        self.i2i_msg = nn.ModuleList()
        for i in range(num_rate_types):
            self.i2i_msg.append(nn.Sequential(
                nn.Linear(self.in_i_chan+self.in_i_chan, int(2*out_i_chan)),
                self.act,
                nn.Linear(int(2*out_i_chan), out_i_chan),
            ))
        
        self.i2i_msg_gate = nn.ModuleList()
        for i in range(num_rate_types):
            self.i2i_msg_gate.append(nn.Sequential(
                nn.Linear(self.in_i_chan+self.in_i_chan, self.num_heads, bias=False),
            ))

        self.attn_gate_ii = nn.Linear(self.in_i_chan, out_i_chan)
        
        # Update species embedding from (self + aggregated reaction effects)
        self.ii_update = nn.Sequential(
            nn.Linear(self.in_i_chan + out_i_chan, int(2*out_i_chan)),
            self.act,
            nn.Linear(int(2*out_i_chan), out_i_chan),
        )

        if self.in_i_chan!=out_i_chan:
            self.ii_proj = nn.Linear(self.in_i_chan,out_i_chan)
        else:
            self.ii_proj = None

        self.in_i_chan = out_i_chan
        
        
        # Messages from iso -> reaction (use reactant species embeddings)
        self.i2r_msg = nn.ModuleList()
        for i in range(num_rate_types):
            self.i2r_msg.append(nn.Sequential(
                nn.Linear(self.in_i_chan+self.in_r_chan, int(2*out_r_chan)),
                self.act, # put this outside of msg, after the msg?
                nn.Linear(int(2*out_r_chan), out_r_chan),
            ))

        self.i2r_msg_gate = nn.ModuleList()
        for i in range(num_rate_types):
            self.i2r_msg_gate.append(nn.Sequential(
                nn.Linear(self.in_i_chan+self.in_r_chan, self.num_heads, bias=False),
            ))

        self.attn_gate_r = nn.Linear(self.in_r_chan, out_r_chan)
        
        # Update reaction embedding from (base + aggregated reactant info)
        self.r_update = nn.ModuleList()
        for i in range(num_rate_types):
            self.r_update.append(nn.Sequential(
                nn.Linear(self.in_r_chan + out_r_chan, int(2*out_r_chan)),
                self.act,
                nn.Linear(int(2*out_r_chan), out_r_chan),
            ))
        
        if self.in_r_chan!=out_r_chan:
            self.r_proj = nn.Linear(self.in_r_chan,out_r_chan)
        else:
            self.r_proj = None

        
        # Messages from reaction -> iso
        self.r2i_msg = nn.ModuleList()
        for i in range(num_rate_types):
            self.r2i_msg.append(nn.Sequential(
                nn.Linear(out_r_chan + self.in_i_chan, int(2*out_i_chan)),
                self.act,
                nn.Linear(int(2*out_i_chan), out_i_chan),
            ))

        self.r2i_msg_gate = nn.ModuleList()
        for i in range(num_rate_types):
            self.r2i_msg_gate.append(nn.Sequential(
                nn.Linear(out_r_chan + self.in_i_chan, self.num_heads, bias=False),
            ))

        self.attn_gate_i = nn.Linear(self.in_i_chan, out_i_chan)
        
        # Update species embedding from (self + aggregated reaction effects)
        self.i_update = nn.Sequential(
            nn.Linear(self.in_i_chan + out_i_chan, int(2*out_i_chan)),
            self.act,
            nn.Linear(int(2*out_i_chan), out_i_chan),
        )


        self.act_attn = nn.LeakyReLU(0.2, inplace=False)

        self.norm_r = nn.LayerNorm(self.reaction_len)
        self.norm_i = nn.LayerNorm(self.iso_len)
        self.norm_ii = nn.LayerNorm(self.iso_len)
            

    def NZ_embedding(self, theinput):
        emb_N = self.emb_N(iso_NZ[:,0])[None,:,:].expand(theinput.shape[0],-1,-1)
        emb_Z = self.emb_Z(iso_NZ[:,1])[None,:,:].expand(theinput.shape[0],-1,-1)

        return torch.cat([theinput, emb_N, emb_Z], dim=-1)

    def forward(self, input_i, input_r):
        """
        input_i: [BS, iso_len, in_i_chan]
        input_r: [BS, reaction_len, in_r_chan]
        """

        if self.NZ_emb_TF:
            input_i = self.NZ_embedding(input_i)
        
        # ----- Reaction base from attributes -----
        if self.r_base_TF:
            parts = []
            for i in range(num_rate_types):
                rea_s = slicing_rea[i]; rea_e = slicing_rea[i+1]
                parts.append(self.r_base[i](torch.cat([self.emb_types(r_type[rea_s:rea_e]).unsqueeze(0).expand(input_r.shape[0], -1, -1), input_r[:,rea_s:rea_e]], dim=-1)))
            input_r = torch.cat(parts,dim=1)

        
        # ----- Phase C: isotopes -> isotopes (reactants+products) -----        
        agg_ii = torch.zeros((input_i.shape[0], self.iso_len, self.out_i_chan), device=input_i.device, dtype=input_i.dtype)
        for i in range(num_rate_types):
            i2i_s = slicing_i2i[i]; i2i_e = slicing_i2i[i+1]
            thefeat = torch.cat([input_i[:,i2i_dst[i2i_s:i2i_e]],input_i[:,i2i_src[i2i_s:i2i_e]]], dim=-1)
            m_i2i = self.i2i_msg[i](thefeat).reshape(input_i.shape[0], -1, self.num_heads, self.d_i) # [BS, M3, out_i_chan]
            m_i2i = self.drop2d_msg(m_i2i)
            score = scatter_softmax(self.act_attn(self.i2i_msg_gate[i](thefeat)), i2i_dst[i2i_s:i2i_e], agg_ii.shape[1]).unsqueeze(-1)
            m_i2i.mul_(score)
            m_i2i = m_i2i.reshape(input_i.shape[0], -1, self.out_i_chan)
            agg_ii.scatter_add_(dim=1, index=i2i_dst[i2i_s:i2i_e].view(1, -1, 1).expand(m_i2i.shape[0],-1,m_i2i.shape[2]), src=m_i2i) # [BS, iso_len, out_i_chan]; sum into species
        agg_ii = self.norm_ii(agg_ii.permute(0,2,1)).permute(0,2,1)

            
        # Update species embeddings (residual)
        agg_ii = self.ii_update(torch.cat([input_i, agg_ii], dim=-1))
        agg_ii = agg_ii*torch.sigmoid(self.attn_gate_ii(input_i))

            
        if self.ii_proj is None:
            input_i = input_i + agg_ii # [BS, iso_len, out_i_chan]
        else:
            input_i = self.ii_proj(input_i) + agg_ii # [BS, iso_len, out_i_chan]
        
        # ----- Phase A: isotopes -> reaction aggregation (reactants) -----       
        agg_r = torch.zeros((input_r.shape[0], self.reaction_len, self.out_r_chan), device=input_i.device, dtype=input_i.dtype)
        for i in range(num_rate_types):
            i2r_s = slicing_i2r[i]; i2r_e = slicing_i2r[i+1]
            thefeat = torch.cat([input_i[:,i2r_src[i2r_s:i2r_e]],input_r[:,i2r_dst[i2r_s:i2r_e]]], dim=-1)
            m_i2r = self.i2r_msg[i](thefeat).reshape(input_r.shape[0], -1, self.num_heads, self.d_r) # [BS, M1, out_r_chan]
            m_i2r = self.drop2d_msg(m_i2r)
            score = scatter_softmax(self.act_attn(self.i2r_msg_gate[i](thefeat)), i2r_dst[i2r_s:i2r_e], agg_r.shape[1]).unsqueeze(-1)
            m_i2r.mul_(score)
            m_i2r = m_i2r.reshape(input_r.shape[0], -1, self.out_r_chan)
            agg_r.scatter_add_(dim=1, index=i2r_dst[i2r_s:i2r_e].view(1, -1, 1).expand(m_i2r.shape[0],-1,m_i2r.shape[2]), src=m_i2r) # [BS, reaction_len, out_r_chan]; sum into reactions
        agg_r = self.norm_r(agg_r.permute(0,2,1)).permute(0,2,1)
        
        # Update reaction embeddings (residual)
        for i in range(num_rate_types):
            rea_s = slicing_rea[i]; rea_e = slicing_rea[i+1]
            agg_r[:,rea_s:rea_e] = self.r_update[i](torch.cat([input_r[:,rea_s:rea_e], agg_r[:,rea_s:rea_e]], dim=-1)) # use input_r or add input_r?
        agg_r = agg_r*torch.sigmoid(self.attn_gate_r(input_r))
                
        if self.r_proj is None:
            input_r = input_r + agg_r # [BS, reaction_len, out_r_chan]
        else:
            input_r = self.r_proj(input_r) + agg_r # [BS, reaction_len, out_r_chan]

        # ----- Phase B: reaction -> isotopes (reactants+products) -----       
        agg_i = torch.zeros((input_i.shape[0], self.iso_len, self.out_i_chan), device=input_i.device, dtype=input_i.dtype)
        for i in range(num_rate_types):
            r2i_s = slicing_r2i[i]; r2i_e = slicing_r2i[i+1]
            thefeat = torch.cat([input_i[:,r2i_dst[r2i_s:r2i_e]],input_r[:,r2i_src[r2i_s:r2i_e]]], dim=-1)
            m_r2i = self.r2i_msg[i](thefeat).reshape(input_i.shape[0], -1, self.num_heads, self.d_i) # [BS, M2, out_i_chan]
            m_r2i = self.drop2d_msg(m_r2i)
            score = scatter_softmax(self.act_attn(self.r2i_msg_gate[i](thefeat)), r2i_dst[r2i_s:r2i_e], agg_i.shape[1]).unsqueeze(-1)
            m_r2i.mul_(score)
            m_r2i = m_r2i.reshape(input_i.shape[0], -1, self.out_i_chan)
            m_r2i.mul_(r2i_coeff[r2i_s:r2i_e][None, :, None]) # apply sign; add this as feature?
            agg_i.scatter_add_(dim=1, index=r2i_dst[r2i_s:r2i_e].view(1, -1, 1).expand(m_r2i.shape[0],-1,m_r2i.shape[2]), src=m_r2i) # [BS, iso_len, out_i_chan]; sum into species
        agg_i = self.norm_i(agg_i.permute(0,2,1)).permute(0,2,1)
        
        
        # Update species embeddings (residual)
        agg_i = self.i_update(torch.cat([input_i, agg_i], dim=-1))
        agg_i = agg_i*torch.sigmoid(self.attn_gate_i(input_i))

        input_i = input_i + agg_i # [BS, iso_len, out_i_chan]

        return input_i, input_r



class FluxPreprocessor(nn.Module):
    def __init__(self, flux_index, in_i_chan, out_i_chan):
        super().__init__()

        self.act = nn.LeakyReLU(inplace=False)
        
        self.flux_index = flux_index                
        
        self.MLP = nn.Sequential(
            nn.Linear(in_i_chan, int(2*out_i_chan)),
            self.act,
            nn.Linear(int(2*out_i_chan), out_i_chan),
        )

        thelen = out_i_chan+1+1+1 # hidden, flux, |flux|, sign(flux)
        
        self.sigmoid_gate = nn.Sequential(
            nn.Linear(thelen, int(2*thelen)),
            self.act,
            nn.Linear(int(2*thelen), 1),
            nn.Sigmoid(),
        )

        self.sigmoid_gate2 = nn.Sequential(
            nn.Linear(thelen, int(2*thelen)),
            self.act,
            nn.Linear(int(2*thelen), 1),
            nn.Sigmoid(),
        )

        self.out_i_chan = in_i_chan#+1+1#+1+1
    
            
    def forward(self, input_i):

        h = self.MLP(input_i)
                
        f = input_i[:,:,self.flux_index:self.flux_index+1]

        thefeat = torch.cat([h,f,torch.abs(f),torch.sign(f)],dim=-1)
        g = self.sigmoid_gate(thefeat)
        g2 = self.sigmoid_gate2(thefeat)

        input_i = input_i.clone() # DON'T NEED THIS DURING INFERENCE
        input_i[:,:,self.flux_index:self.flux_index+1] = g-g2+f
        
        return input_i


class PureGNN_Final(nn.Module):
    def __init__(self):
        super().__init__()
            
        self.num_layers = param_dic_GNN['num_layers']
                
        out_i_chan0 = param_dic_GNN['chan']
        out_r_chan0 = param_dic_GNN['chan']
        out_i_chans = [out_i_chan0 for i in range(self.num_layers)]
        out_r_chans = [out_r_chan0 for i in range(self.num_layers)]
        
        self.preprocessor = FluxPreprocessor(1, in_channels_iso_len, out_i_chan0)
        
        self.layers = nn.ModuleList([])
        self.layers.append(MPNNLayer_GAT(self.preprocessor.out_i_chan, in_channels_rea_len, out_i_chans[0], out_r_chans[0], NZ_emb_TF = True, r_base_TF = True))
        for i in range(self.num_layers-2):
            self.layers.append(MPNNLayer_GAT(out_i_chans[i], out_r_chans[i], out_i_chans[i+1], out_r_chans[i+1], NZ_emb_TF = False, r_base_TF = False))
        
        self.layers.append(nn.Linear(out_i_chans[self.num_layers-2], 2))
        
        # per layer, FiLM, MLP(input_1, input_i, input_r), sigmoid constrained
        self.diffusions_FiLM = FiLM_layer(self.num_layers-2)
        
    def forward(self, input_i, input_r):

        diffusions = torch.sigmoid(self.diffusions_FiLM(input_i[:,0:1,-3:]))

        input_i = self.preprocessor(input_i)
        
        d=0
        for layer in self.layers[:-1]:
            input_i, input_r = layer(input_i, input_r)
            if d==0:
                input_i0 = input_i
            else:
                alpha = diffusions[:,:,d-1:d]
                input_i = (1-alpha)*input_i+alpha*input_i0
            d+=1
            
        out = torch.sigmoid(self.layers[-1](input_i)) # add original input_i too by cat?
        
        out = out[:,:,0:1] - out[:,:,1:2]
        
        return out

