import os
import sys
source=str(sys.argv[1])
target=str(sys.argv[2])
data_name_or_path=str(sys.argv[3])
filen=sys.argv[4]
#'/private/home/costajussa/interpretability/nmt/data/'

print (data_name_or_path)

import torch
#torch.cuda.set_device(1)
#torch.cuda.current_device()

import warnings
from pathlib import Path

from wrappers.multilingual_transformer_wrapper import FairseqMultilingualTransformerHub

from fairseq.data.multilingual.multilingual_utils import (
    EncoderLangtok,
    LangTokSpec,
    LangTokStyle,
    augment_dictionary,
    get_lang_tok,
)

import alignment.align as align

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
import fairseq

import logging
logger = logging.getLogger()
logger.setLevel('WARNING')
#warnings.simplefilter('ignore')

from dotenv import load_dotenv
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

model_size = 'small' # small (412M) /big (1.2B)
data_sample = 'generate' # generate/interactive
teacher_forcing = False # teacher forcing/free decoding


# Paths
# Checkpoint path
ckpt_dir = '/checkpoint/vedanuj/nmt/flores200/dense_dae_ssl.mfp16.mu100000.uf2.lss.tmp1.0.lr0.001.drop0.1.maxtok5120.seed2.max_pos512.shem.NBF.adam16bit.fully_sharded.enttgt.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E2048.H16.ATTDRP0.1.RELDRP0.0.m0.3.mr0.1.i0.0.p0.0.r0.0.pl3.5.rl1.ngpu128'

# Path to binarized data

checkpoint_file = 'checkpoint_4_100000-shard0.pt'

hub = FairseqMultilingualTransformerHub.from_pretrained(
    ckpt_dir,
    checkpoint_file=checkpoint_file,
    data_name_or_path=data_name_or_path,
    dict_path='/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.4.256k/sentencepiece.source.256000.source.dict.txt',
    source_lang= source,
    target_lang= target,
    lang_pairs =source+'-'+target)
NUM_LAYERS = 24

#print('hub',len(hub.task.target_dictionary))
if data_sample=='generate':
    with open(data_name_or_path+str(filen)+"."+source, 'r') as fp:
        for i, line in enumerate(fp):    
            #print (i,line)
            src_sent, src_tok, src_tensor, tgt_sent, tgt_tok, tgt_tensor = hub.get_sample(filen, i)
            #print (src_tensor)
            #print (tgt_tensor)
            src_lan_token = get_lang_tok(lang=hub.task.source_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
            tgt_lan_token = get_lang_tok(lang=hub.task.target_langs[0], lang_tok_style=LangTokStyle.multilingual.value)

            if teacher_forcing:
                #print ('srctensor',src_tensor.size())
                #print ('tgttensor',tgt_tensor.size())
                model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)

                #print("\n\nGREEDY DECODING\n")
                pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
                pred_tok = hub.decode(pred_tensor, hub.task.target_dictionary)
                pred_sent = hub.decode(pred_tensor, hub.task.target_dictionary, as_string=True)
                #print(f"Predicted sentence: \t {pred_sent}")
                target_sentence = ['</s>'] + [tgt_lan_token] + tgt_tok
                
            if not teacher_forcing: #we do not need a reference here
                tgt_tensor_free = []
                print(src_tensor)
                #print("\n\nBEAM SEARCH\n")
                src_tensor = src_tensor[1:]
                for pred in hub.generate(src_tensor,4,verbose=True): #added 0
                    tgt_tensor_free.append(pred['tokens'])
                    pred_sent = hub.decode(pred['tokens'], hub.task.target_dictionary, as_string=True)
                    score = pred['score'].item()
                    #print(f"{score} \t {pred_sent}")

                    hypo = 0# first hypothesis we do teacher forcing with the best hypothesis
                    tgt_tensor = tgt_tensor_free[hypo]
    
                    # We add eos token at the beginning of sentence and delete it from the end
                    tgt_tensor = torch.cat([torch.tensor([hub.task.target_dictionary.eos_index]).to(tgt_tensor.device),tgt_tensor[:-1]]).to(tgt_tensor.device)
                    tgt_tok = hub.decode(tgt_tensor, hub.task.target_dictionary, as_string=False)
                    target_sentence = tgt_tok
                    pred_tok = tgt_tok

                  #  model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)

                    #print(f"\n\nGREEDY DECODING with hypothesis {hypo+1}\n")
                #    pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
                #    pred_tok = hub.decode(pred_tensor, hub.task.target_dictionary)
                #    pred_sent = hub.decode(pred_tensor, hub.task.target_dictionary, as_string=True)
                 #   print(f"Predicted sentence: \t {pred_sent}")
                    #break

                source_sentence = src_tok
                target_sentence = tgt_tok
                predicted_sentence = pred_tok

                #relevances_enc_self_attn = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1', norm_modese='min_sum', pre_layer_norm=True)['encoder.self_attn']
            #print(src_tensor.size())
            #print(tgt_tensor.size())
            enc_self_attn_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum', pre_layer_norm=True)['encoder.self_attn'])                                            
            #print('enc_self')
            total_rollout = hub.get_contribution_rollout(src_tensor, tgt_tensor,
                                            'l1', norm_mode='min_sum',
                                            pre_layer_norm=True)['total']


            for layer in range(0,NUM_LAYERS):

                contributions_rollout_layer = total_rollout[layer]
                contributions_rollout_layer_np = contributions_rollout_layer.detach().cpu().numpy()
                    #    df = pd.DataFrame(contributions_rollout_layer_np, columns = source_sentence + target_sentence, index = predicted_sentence)
                src_contribution = contributions_rollout_layer_np[:,:len(src_tok)].sum(-1) #src contribution for target token, 
                src_contribution_mean= np.mean (src_contribution)
                    #trg_contribution_mean= 1-src_contribution

            print (src_contribution_mean)


