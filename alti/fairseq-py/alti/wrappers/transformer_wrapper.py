import logging
import warnings
from functools import partial
from collections import defaultdict

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer import TransformerModel

from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns

rc={'font.size': 12, 'axes.labelsize': 10, 'legend.fontsize': 10.0,
    'axes.titlesize': 24, 'xtick.labelsize': 24, 'ytick.labelsize': 24,
    'axes.linewidth': .5, 'figure.figsize': (12,12)}
plt.rcParams.update(**rc)


class FairseqTransformerHub(GeneratorHubInterface):
    ATTN_MODULES = ['encoder.self_attn',
                    'decoder.self_attn',
                    'decoder.encoder_attn']

    def __init__(self, cfg, task, models):
        super().__init__(cfg, task, models,add_lang_tok=False)
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def from_pretrained(cls, checkpoint_dir, checkpoint_file, data_name_or_path):
        hub_interface = TransformerModel.from_pretrained(checkpoint_dir, checkpoint_file, data_name_or_path)
        return cls(hub_interface.cfg, hub_interface.task, hub_interface.models)
    
    def encode(self, sentence, dictionary):
        raise NotImplementedError()
    
    def decode(self, tensor, dictionary, as_string=False):
        tok = dictionary.string(tensor).split()
        if as_string:
            return ''.join(tok).replace('▁', ' ')
        else:
            return tok
    
    def get_sample(self, split, index):

        if split not in self.task.datasets.keys():
            print(split)
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]['source']
        src_tok = self.decode(src_tensor, self.task.src_dict)
        src_sent = self.decode(src_tensor, self.task.src_dict, as_string=True)

        tgt_tensor = self.task.dataset(split)[index]['target']
        tgt_tok = self.decode(tgt_tensor, self.task.tgt_dict)
        tgt_sent = self.decode(tgt_tensor, self.task.tgt_dict, as_string=True)
        
    
        return src_sent, src_tok, src_tensor, tgt_sent, tgt_tok, tgt_tensor
   
    def parse_module_name(self, module_name):
        """ Returns (enc_dec, layer, module)"""
        parsed_module_name = module_name.split('.')
        if not isinstance(parsed_module_name, list):
            parsed_module_name = [parsed_module_name]
            
        if len(parsed_module_name) < 1 or len(parsed_module_name) > 3:
            raise AttributeError(f"'{module_name}' unknown")
            
        if len(parsed_module_name) > 1:
            try:
                parsed_module_name[1] = int(parsed_module_name[1])
            except ValueError:
                parsed_module_name.insert(1, None)
            if len(parsed_module_name) < 3:
                parsed_module_name.append(None)
        else:
            parsed_module_name.extend([None, None])

        return parsed_module_name
    
    def get_module(self, module_name):
        e_d, l, m = self.parse_module_name(module_name)
        module = getattr(self.models[0], e_d)
        if l is not None:
            module = module.layers[l]
            if m is not None:
                module = getattr(module, m)
        else:
            if m is not None:
                raise AttributeError(f"Cannot get'{module_name}'")

        return module

    def trace_forward(self, src_tensor, tgt_tensor):
        r"""Forward-pass through the model.
        Args:
            src_tensor (`tensor`):
                Source sentence tensor.
            tgt_tensor (`tensor`):
                Target sentence tensor (teacher forcing).
        Returns:
            model_output ('tuple'):
                output of the model.
            log_probs:
                log probabilities output by the model.
            encoder_output ('dict'):
                dictionary with 'encoder_out', 'encoder_padding_mask', 'encoder_embedding',
                                'encoder_states', 'src_tokens', 'src_lengths', 'attn_weights'.
            layer_inputs:
                dictionary with the input of the modeules of the model.
            layer_outputs:
                dictionary with the input of the modeules of the model.
        """
        self.zero_grad()

        layer_inputs = defaultdict(list)
        layer_outputs = defaultdict(list)

        def save_activation(name, mod, inp, out):
            layer_inputs[name].append(inp)
            layer_outputs[name].append(out)

        handles = {}

        for name, layer in self.named_modules():
            handles[name] = layer.register_forward_hook(partial(save_activation, name))
        
        src_tensor = src_tensor.unsqueeze(0).to(self.device)
        tgt_tensor = tgt_tensor.unsqueeze(0).to(self.device)

        model_output, encoder_out = self.models[0](src_tensor, src_tensor.size(-1), tgt_tensor, )

        log_probs = self.models[0].get_normalized_probs(model_output, log_probs=True, sample=None)
        
        for k, v in handles.items():
            handles[k].remove()
        
        return model_output, log_probs, encoder_out, layer_inputs, layer_outputs

    def normalize_contrib(self, x, mode=None, temperature=0.5, resultant_norm=None):
        """ Normalization applied to each row of the layer-wise contributions."""
        if mode == 'min_max':
            # Min-max normalization
            x_min = x.min(-1, keepdim=True)[0]
            x_max = x.max(-1, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min)
            x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
        elif mode == 'softmax':
            # Softmax
            x_norm = F.softmax(x / temperature, dim=-1)
        elif mode == 'sum_one':
            # Sum one
            x_norm = x / x.sum(dim=-1, keepdim=True)
        elif mode == 'min_sum':
            # Minimum value selection
            if resultant_norm == None:
                x_min = x.min(-1, keepdim=True)[0]
                x_norm = x + torch.abs(x_min)
                x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
            else:
                x_norm = x + torch.abs(resultant_norm.unsqueeze(1))
                x_norm = torch.clip(x_norm,min=0)
                x_norm = x_norm / x_norm.sum(dim=-1,keepdim=True)
        elif mode is None:
            x_norm = x
        else:
            raise AttributeError(f"Unknown normalization mode '{mode}'")
        return x_norm

    def __get_attn_weights_module(self, layer_outputs, module_name):
        enc_dec_, l, attn_module_ = self.parse_module_name(module_name)
        
        attn_module = self.get_module(module_name)
        num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim

        k = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.k_proj"][0]
        q = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0]

        q, k = map(
            lambda x: rearrange(
                x,
                't b (n_h h_d) -> (b n_h) t h_d',
                n_h=num_heads,
                h_d=head_dim
            ),
            (q, k)
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if enc_dec_ == 'decoder' and attn_module_ == 'self_attn':
            tri_mask = torch.triu(torch.ones_like(attn_weights), 1).bool()
            attn_weights[tri_mask] = -1e9

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = rearrange(
            attn_weights,
            '(b n_h) t_q t_k -> b n_h t_q t_k',
            n_h=num_heads
        )
        return attn_weights
    
    def __get_contributions_module(self, layer_inputs, layer_outputs, contrib_type, pre_layer_norm, module_name):
        enc_dec_, l, attn_module_ = self.parse_module_name(module_name)
        attn_w = self.__get_attn_weights_module(layer_outputs, module_name) # (batch_size, num_heads, src:len, src_len)
        
        def l_transform(x, w_ln):
            '''Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation.'''
            ln_param_transf = torch.diag(w_ln)
            ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - \
                1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)

            out = torch.einsum(
                '... e , e f , f g -> ... g',
                x,
                ln_mean_transf,
                ln_param_transf
            )
            return out

        attn_module = self.get_module(module_name)
        w_o = attn_module.out_proj.weight
        b_o = attn_module.out_proj.bias
        
        ln = self.get_module(f'{module_name}_layer_norm')
        w_ln = ln.weight.data
        b_ln = ln.bias
        eps_ln = ln.eps
        
        in_q = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0][0].transpose(0, 1)
        in_v = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.v_proj"][0][0].transpose(0, 1)
        in_res = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)

        if "self_attn" in attn_module_:
            if pre_layer_norm:
                residual_ = torch.einsum('sk,bsd->bskd', torch.eye(in_res.size(1)).to(in_res.device), in_res)
            else:
                residual_ = torch.einsum('sk,bsd->bskd', torch.eye(in_q.size(1)).to(in_res.device), in_q)
        else:
            if pre_layer_norm:
                residual_ = in_res
            else:
                residual_ = in_q

        v = attn_module.v_proj(in_v)
        v = rearrange(
            v,
            'b t_v (n_h h_d) -> b n_h t_v h_d',
            n_h=attn_module.num_heads,
            h_d=attn_module.head_dim
        )

        w_o = rearrange(
            w_o,
            'out_d (n_h h_d) -> n_h h_d out_d',
            n_h=attn_module.num_heads,
        )

        attn_v_wo = torch.einsum(
            'b h q k , b h k e , h e f -> b q k f',
            attn_w,
            v,
            w_o
        )

        # Add residual
        if "self_attn" in attn_module_:
            out_qv_pre_ln = attn_v_wo + residual_
        # Concatenate residual in cross-attention (as another value vector)
        else:
            out_qv_pre_ln = torch.cat((attn_v_wo,residual_.unsqueeze(-2)),dim=2)
        
        # Assert MHA output + residual is equal to pre-layernorm input
        out_q_pre_ln = out_qv_pre_ln.sum(-2) +  b_o

        if pre_layer_norm:
            if 'encoder' in enc_dec_:
                 # Encoder (delf-attention) -> final_layer_norm
                out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
                
            else:
                if "self_attn" in attn_module_:
                    # Self-attention decoder -> encoder_attn_layer_norm
                    out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.encoder_attn_layer_norm"][0][0].transpose(0, 1)
                else:
                    # Cross-attention decoder -> final_layer_norm
                    out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
                
        else:
            # In post-ln we compare with the input of the first layernorm
            out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)

        # print('out_q_pre_ln_th',out_q_pre_ln_th[0,0,:10])
        # print('out_q_pre_ln',out_q_pre_ln[0,0,:10])
        assert torch.dist(out_q_pre_ln_th, out_q_pre_ln).item() < 1e-3 * out_q_pre_ln.numel()
        
        if pre_layer_norm:
            transformed_vectors = out_qv_pre_ln
            resultant = out_q_pre_ln
        else:
            ln_std_coef = 1/(out_q_pre_ln_th + eps_ln).std(-1).view(1,-1, 1).unsqueeze(-1) # (batch,src_len,1,1)
            transformed_vectors = l_transform(out_qv_pre_ln, w_ln)*ln_std_coef # (batch,src_len,tgt_len,embed_dim)
            dense_bias_term = l_transform(b_o, w_ln)*ln_std_coef # (batch,src_len,1,embed_dim)
            attn_output = transformed_vectors.sum(dim=2) # (batch,seq_len,embed_dim)
            resultant = attn_output + dense_bias_term.squeeze(2) + b_ln # (batch,seq_len,embed_dim)   

            # Assert resultant (decomposed attention block output) is equal to the real attention block output
            out_q_th_2 = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0].transpose(0, 1)
            assert torch.dist(out_q_th_2, resultant).item() < 1e-3 * resultant.numel()
        #print (transformed_vectors.size(),resultant.size())
        if contrib_type == 'l1':
            contributions = -F.pairwise_distance(transformed_vectors, resultant.unsqueeze(2), p=1)
            resultants_norm = torch.norm(torch.squeeze(resultant),p=1,dim=-1)
        elif contrib_type == 'l2':
            contributions = -F.pairwise_distance(transformed_vectors, resultant.unsqueeze(2), p=2)
        else:
            raise ArgumentError(f"contribution_type '{contrib_type}' unknown")

        #print ('getcontrmodule',contributions.size(),resultants_norm.size())
        return contributions, resultants_norm
    
    def get_contributions(self, src_tensor, tgt_tensor, contrib_type='l1', norm_mode='min_sum', pre_layer_norm=False):
        r"""
        Get contributions for each ATTN_MODULE: 'encoder.self_attn', 'decoder.self_attn', 'decoder.encoder_attn.
        Args:
            src_tensor (`tensor` ()):
                Source sentence tensor.
            tgt_tensor (`tensor` ()):
                Target sentence tensor (teacher forcing).
            contrib_type (`str`, defaults to `l1` (Ferrando et al ., 2022)):
                Type of layer-wise contribution measure: l1, l2, or attn_w.
            norm_mode ('str', defaults to `min_sum` (Ferrando et al ., 2022)):
                Type of normalization applied to layer-wise contributions: 'min_sum', 'min_max', 'sum_one', 'softmax'.
        Returns:
            Dictionary with ATTN_MODULE as keys, and tensor with contributions (batch_size, num_layers, src_len, tgt_len) as values.
        """
        contributions_all = defaultdict(list)
        _, _, _, layer_inputs, layer_outputs = self.trace_forward(src_tensor, tgt_tensor)
        
        if contrib_type == 'attn_w':
            f = partial(self.__get_attn_weights_module, layer_outputs)
        else:
            f = partial(
                self.__get_contributions_module,
                layer_inputs,
                layer_outputs,
                contrib_type,
                pre_layer_norm
            )

        for attn in self.ATTN_MODULES:
            enc_dec_, _, attn_module_ = self.parse_module_name(attn)
            enc_dec = self.get_module(enc_dec_)
            #print('enc_dec',enc_dec_)
            #print('attn',attn)
            #print('getattnsrctensor',src_tensor.size())
            #print('getattntrgtensor',tgt_tensor.size())

            for l in range(len(enc_dec.layers)):
                contributions, resultant_norms = f(attn.replace('.', f'.{l}.'))
                #print ('csize',contributions.size())
                #print('rnorm',resultant_norms.size())
                contributions = self.normalize_contrib(contributions, norm_mode, resultant_norm=resultant_norms).unsqueeze(1)
                # Mask upper triangle of decoder self-attention matrix (and normalize)
                # if attn == 'decoder.self_attn':
                #     contributions = torch.tril(torch.squeeze(contributions,dim=1))
                #     contributions = contributions / contributions.sum(dim=-1, keepdim=True)
                #     contributions = contributions.unsqueeze(1)
                contributions_all[attn].append(contributions)
        contributions_all = {k: torch.cat(v, dim=1) for k, v in contributions_all.items()}
        return contributions_all

    def get_contribution_rollout(self, src_tensor, tgt_tensor, contrib_type='l1', norm_mode='min_sum', pre_layer_norm=False, **contrib_kwargs):
        # c = self.get_contributions(src_tensor, tgt_tensor, contrib_type, norm_mode, **contrib_kwargs)
        # if contrib_type == 'attn_w':
        #     c = {k: v.sum(2) for k, v in c.items()}
        
        def compute_joint_attention(att_mat):
            """ Compute attention rollout given contributions or attn weights + residual."""
            joint_attentions = torch.zeros(att_mat.size()).to(att_mat.device)
            layers = joint_attentions.shape[0]
            joint_attentions = att_mat[0].unsqueeze(0)
            for i in range(1,layers):
                C_roll_new = torch.matmul(att_mat[i],joint_attentions[i-1])
                joint_attentions = torch.cat([joint_attentions, C_roll_new.unsqueeze(0)], dim=0)
            return joint_attentions
        
        # Rollout encoder
    #    def compute_joint_attention(att_mat):
    #        """ Compute attention rollout given contributions or attn weights + residual."""

    #        aug_att_mat =  att_mat
    #        device = att_mat.device
    #        joint_attentions = torch.zeros(aug_att_mat.size()).to(device)

#            layers = joint_attentions.shape[0]
 #           joint_attentions[0] = aug_att_mat[0]
  #          
             #for i in range(1,layers):
   #             joint_attentions[i] = torch.matmul(aug_att_mat[i],joint_attentions[i-1])
                
   #         return joint_attentions

        c_roll = defaultdict(list)
        enc_sa = 'encoder.self_attn'

        # Compute contributions rollout encoder self-attn
        enc_self_attn_contributions = torch.squeeze(self.get_contributions(src_tensor, tgt_tensor, contrib_type, norm_mode=norm_mode, pre_layer_norm=pre_layer_norm)[enc_sa])
        layers, _, _ = enc_self_attn_contributions.size()
        enc_self_attn_contributions_mix = compute_joint_attention(enc_self_attn_contributions)
        c_roll[enc_sa] = enc_self_attn_contributions_mix

        # Get last layer relevances w.r.t input
        relevances_enc_self_attn = enc_self_attn_contributions_mix[-1]
        # repeat num_layers times
        relevances_enc_self_attn = relevances_enc_self_attn.unsqueeze(0).repeat(layers, 1, 1)
            
        def rollout(C, C_enc_out):
            """ Contributions rollout whole Transformer-NMT model.
                Args:
                    C: [cross_attn;self_dec_attn] before encoder rollout
                    C_enc_out: encoder rollout last layer
            """
            src_len = C.size(2) - C.size(1)
            tgt_len = C.size(1)

            C_sa_roll = C[:, :, -tgt_len:]     # Self-att, only has 1 layer (last)
            C_ed_roll = torch.einsum(          # encoder rollout*cross-attn
                "lie , ef -> lif",
                C[:, :, :src_len],             # Cross-att
                C_enc_out                      # Encoder rollout
            )

            C_roll = torch.cat([C_ed_roll, C_sa_roll], dim=-1) # [(cross_attn*encoder rollout);self_dec_attn]
            C_roll_new_accum = C_roll[0].unsqueeze(0)

            for i in range(1, len(C)):
                C_sa_roll_new = torch.einsum(
                    "ij , jk -> ik",
                    C_roll[i, :, -tgt_len:],   # Self-att dec
                    C_roll_new_accum[i-1, :, -tgt_len:], # Self-att (prev. roll)
                )
                C_ed_roll_new = torch.einsum(
                    "ij , jk -> ik",
                    C_roll[i, :, -tgt_len:],  # Self-att dec
                    C_roll_new_accum[i-1, :, :src_len], # Cross-att (prev. roll)
                ) + C_roll[i, :, :src_len]    # Cross-att

                C_roll_new = torch.cat([C_ed_roll_new, C_sa_roll_new], dim=-1)
                C_roll_new = C_roll_new / C_roll_new.sum(dim=-1,keepdim=True)
                
                C_roll_new_accum = torch.cat([C_roll_new_accum, C_roll_new.unsqueeze(0)], dim=0)
                

            return C_roll_new_accum

        dec_sa = 'decoder.self_attn'
        dec_ed = 'decoder.encoder_attn'
        
        # Compute joint cross + self attention
        self_dec_contributions = torch.squeeze(self.get_contributions(src_tensor, tgt_tensor, contrib_type, norm_mode=norm_mode, pre_layer_norm=pre_layer_norm)[dec_sa])
        cross_contributions = torch.squeeze(self.get_contributions(src_tensor, tgt_tensor, contrib_type, norm_mode=norm_mode, pre_layer_norm=pre_layer_norm)[dec_ed])
        self_dec_contributions = (self_dec_contributions.transpose(1,2)*cross_contributions[:,:,-1].unsqueeze(1)).transpose(1,2)
        joint_self_cross_contributions = torch.cat((cross_contributions[:,:,:-1],self_dec_contributions),dim=-1)

        contributions_full_rollout = rollout(joint_self_cross_contributions, relevances_enc_self_attn[-1])
        c_roll['total'] = contributions_full_rollout

        return c_roll
    
    # def viz_contributions(self, src_tensor, tgt_tensor, contrib_type, roll=False, attn=None, layer=None, head=None, **contrib_kwargs):
    #     if roll:
    #         contrib = self.get_contribution_rollout(src_tensor, tgt_tensor, contrib_type, **contrib_kwargs)
    #     else:
    #         contrib = self.get_contributions(src_tensor, tgt_tensor, contrib_type, **contrib_kwargs)
        
    #     src_tok = self.decode(src_tensor, self.task.src_dict)
    #     tgt_tok = self.decode(tgt_tensor, self.task.tgt_dict)
        
    #     def what_to_show(arg, valid_values):
    #         valid_type = type(valid_values[0])
    #         if arg is None:
    #             to_show = valid_values
    #         elif isinstance(arg, valid_type):
    #             to_show = [arg]
    #         elif isinstance(arg, list):
    #             to_show = [a for a in arg if a in valid_values]
    #         else:
    #             raise TypeError("Argument must be str, List[str] or None")

    #         return to_show
        
    #     def show_contrib_heatmap(data, k_tok, q_tok, title):
    #         df = pd.DataFrame(
    #             data=data,
    #             columns=k_tok,
    #             index=q_tok
    #         )

    #         fig, ax = plt.subplots()
    #         g = sns.heatmap(df, cmap="Blues", cbar=True, square=True, ax=ax, fmt='.2f')
    #         g.set_title(title)
    #         g.set_xlabel("Key")
    #         g.set_ylabel("Query")
    #         g.set_xticklabels(g.get_xticklabels(), rotation=50, horizontalalignment='center',fontsize=10)
    #         g.set_yticklabels(g.get_yticklabels(),fontsize=10);

    #         fig.show()  

    #     for a in what_to_show(attn, self.ATTN_MODULES):
    #         enc_dec_, _, attn_module_ = self.parse_module_name(a)
    #         num_layers = self.get_module(enc_dec_).num_layers
    #         if a == 'encoder.self_attn':
    #             q_tok = src_tok + ['<EOS>']
    #             k_tok = src_tok + ['<EOS>']
    #         elif a == 'decoder.self_attn':
    #             q_tok = tgt_tok + ['<EOS>']
    #             k_tok = ['<EOS>'] + tgt_tok
    #         elif a == 'decoder.encoder_attn':
    #             q_tok = tgt_tok + ['<EOS>']
    #             k_tok = src_tok + ['<EOS>']
    #         else:
    #             pass
            
    #         #q_tok = (src_tok + ['<EOS>']) if a == 'encoder.self_attn' else (['<EOS>'] + tgt_tok)
    #         #k_tok = (['<EOS>'] + tgt_tok) if a == 'decoder.self_attn' else (src_tok + ['<EOS>'])

    #         for l in what_to_show(layer, list(range(num_layers))):
    #             num_heads = self.get_module(a.replace('.', f'.{l}.')).num_heads
                
    #             contrib_ = contrib[a][0,l]


    #             if contrib_type == 'attn_w' and roll == False:
    #                 for h in what_to_show(head, [-1] + list(range(num_heads))):
    #                     contrib__ = contrib_.mean(0) if h == -1 else contrib_[h]
    #                     show_contrib_heatmap(
    #                         contrib__.cpu().detach().numpy(), #3
    #                         k_tok,
    #                         q_tok,
    #                         title=f"{contrib_type}; {a}; layer: {l}; head: {'mean' if h == -1 else h}"
    #                     )
    #             else:
    #                 show_contrib_heatmap(
    #                     contrib_.cpu().detach().numpy(),
    #                     k_tok,
    #                     q_tok,
    #                     title=f"{contrib_type}; {a}; layer: {l}"
    #                 )