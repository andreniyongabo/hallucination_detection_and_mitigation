

import os
import sys
source=str(sys.argv[1])
target=str(sys.argv[2])


import torch
torch.cuda.set_device(1)
torch.cuda.current_device()

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
teacher_forcing = True # teacher forcing/free decoding


# Paths
# Checkpoint path
ckpt_dir = '/large_experiments/nllb/mmt/h2_21_models/flores125_v3.3/en_to_many_to_en/v3.3_dense_hrft004.mfp16.mu100000.uf4.lss.enttgt.tmp1.0.shem.NBF.warmup8000.lr0.004.drop0.0.maxtok2560.seed2.valevery200000.max_pos512.adam16bit.fully_sharded.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.ATTDRP0.1.RELDRP0.0.ngpu128/'

# Path to binarized data
testing-interactive/jigsaw-corpus-test/val.eng
data_name_or_path='/private/home/costajussa/
interpretability/nmt/data/'
checkpoint_file = 'checkpoint_15_100000_consolidated.pt'

hub = FairseqMultilingualTransformerHub.from_pretrained(
    ckpt_dir,
    checkpoint_file=checkpoint_file,
    data_name_or_path=data_name_or_path,
    dict_path='/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/data_bin/shard000//dict.eng.txt',
    source_lang= source,
    target_lang= target,
    lang_pairs =source+'-'+target)
NUM_LAYERS = 24


if data_sample=='generate':
    with open(r"./data/devtest-flores"+source+"."+source, 'r') as fp:
        for i, line in enumerate(fp):    
            
            src_sent, src_tok, src_tensor, tgt_sent, tgt_tok, tgt_tensor = hub.get_sample('devtest-flores'+source, i)
            src_lan_token = get_lang_tok(lang=hub.task.source_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
            tgt_lan_token = get_lang_tok(lang=hub.task.target_langs[0], lang_tok_style=LangTokStyle.multilingual.value)

            if teacher_forcing:
                model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)

                #print("\n\nGREEDY DECODING\n")
                pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
                pred_tok = hub.decode(pred_tensor, hub.task.target_dictionary)
                pred_sent = hub.decode(pred_tensor, hub.task.target_dictionary, as_string=True)
                #print(f"Predicted sentence: \t {pred_sent}")
                target_sentence = ['</s>'] + [tgt_lan_token] + tgt_tok

                source_sentence = src_tok
                target_sentence = tgt_tok
                predicted_sentence = pred_tok

                #relevances_enc_self_attn = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1', norm_modese='min_sum', pre_layer_norm=True)['encoder.self_attn']
                                                        
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


