import os
import sys
import torch
import torch.nn as nn

import warnings
from pathlib import Path

from wrappers.multilingual_transformer_wrapper import FairseqMultilingualTransformerHub

import fairseq
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq_cli.interactive import make_batches
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

import logging
logger = logging.getLogger()
logger.setLevel('WARNING')

from dotenv import load_dotenv
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

source=str(sys.argv[1])
target=str(sys.argv[2])
data_name_or_path=str(sys.argv[3])
filen=sys.argv[4]
save_path=sys.argv[5]

model_size = 'small' # small (412M) /big (1.2B)
data_sample = 'generate' # generate/interactive
teacher_forcing = False # teacher forcing/free decoding
input_source = "translated" # translated/interactive

def prepare_input_encoder(hub, tok_sentence):
    generator = hub.task.build_generator(hub.models, hub.cfg)
    max_positions = utils.resolve_max_positions(
    hub.task.max_positions(), *[model.max_positions() for model in hub.models]
    )
    def encode_fn(x):
        return x
    batch = make_batches(tok_sentence,hub.cfg,hub.task,max_positions,encode_fn)
    src_tensor = next(batch).src_tokens
    src_tok = [hub.task.target_dictionary[t] for t in src_tensor[0]]
    return src_tok, src_tensor[0] # first element in batch

def prepare_input_decoder(hub, tok_sentence):
    tok_sentence = tok_sentence.split()
    lang = hub.task.args.langtoks["main"][1]
    if lang == 'tgt':
        lang_tok = hub.task.args.target_lang
    else:
        lang_tok = hub.task.args.source_lang
    lang_tok = get_lang_tok(lang=lang_tok, lang_tok_style=LangTokStyle.multilingual.value)
    tgt_tok = [hub.task.target_dictionary[hub.task.target_dictionary.eos_index]] + [lang_tok] + tok_sentence
    tgt_tensor = torch.tensor([hub.task.target_dictionary.index(t) for t in tgt_tok])
    return tgt_tok, tgt_tensor

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

if data_sample=='generate':
    with open(data_name_or_path+str(filen)+"."+source, 'r') as fp:
        src_contr = []
        for i, line in enumerate(fp):
            if input_source=='interactive':
                interactive_sample = hub.get_interactive_sample(i, data_name_or_path,
                                                                source, target, filen,
                                                                prepare_input_encoder,
                                                                prepare_input_decoder)
                src_tensor = interactive_sample['src_tensor']
                tgt_tensor = interactive_sample['tgt_tensor']
                src_tok = interactive_sample['src_tok']
                tgt_tok = interactive_sample['tgt_tok']

            if input_source=='translated':
                src_sent, src_tok, src_tensor, tgt_sent, tgt_tok, tgt_tensor = hub.get_sample(filen, i)
                src_lan_token = get_lang_tok(lang=hub.task.source_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
                tgt_lan_token = get_lang_tok(lang=hub.task.target_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
                
                if teacher_forcing:
                    model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)

                    #print("\n\nGREEDY DECODING\n")
                    pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
                    pred_tok = hub.decode(pred_tensor, hub.task.target_dictionary)
                    pred_sent = hub.decode(pred_tensor, hub.task.target_dictionary, as_string=True)
                    target_sentence = ['</s>'] + [tgt_lan_token] + tgt_tok
                    
                if not teacher_forcing: #we do not need a reference here
                    tgt_tensor_free = []

                    #print("\n\nBEAM SEARCH\n")
                    src_tensor = src_tensor[1:]
                    for pred in hub.generate(src_tensor,4,verbose=True): #added 0
                        tgt_tensor_free.append(pred['tokens'])
                        pred_sent = hub.decode(pred['tokens'], hub.task.target_dictionary, as_string=True)
                        score = pred['score'].item()

                        hypo = 0 # first hypothesis we do teacher forcing with the best hypothesis
                        tgt_tensor = tgt_tensor_free[hypo]
        
                        # We add eos token at the beginning of sentence and delete it from the end
                        tgt_tensor = torch.cat([torch.tensor([hub.task.target_dictionary.eos_index]).to(tgt_tensor.device),tgt_tensor[:-1]]).to(tgt_tensor.device)
                        tgt_tok = hub.decode(tgt_tensor, hub.task.target_dictionary, as_string=False)
                        target_sentence = tgt_tok
                        pred_tok = tgt_tok

                    source_sentence = src_tok
                    target_sentence = tgt_tok
                    predicted_sentence = pred_tok
        
            enc_self_attn_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum', pre_layer_norm=True)['encoder.self_attn'])                                            

            total_rollout = hub.get_contribution_rollout(src_tensor, tgt_tensor,
                                            'l1', norm_mode='min_sum',
                                            pre_layer_norm=True)['total']

            for layer in range(0,NUM_LAYERS):

                contributions_rollout_layer = total_rollout[layer]
                contributions_rollout_layer_np = contributions_rollout_layer.detach().cpu().numpy()
                src_contribution = contributions_rollout_layer_np[:,:len(src_tok)].sum(-1) #src contribution for target token, 
                src_contribution_mean= np.mean (src_contribution)
                trg_contribution_mean= 1-src_contribution
            
            src_contr.append(src_contribution_mean)

        with open(save_path + "_" + input_source + ".alti_score", "w") as outfile:
            sent_level_score = 0
            for j, score in enumerate(src_contr):
                print(j)
                sent_level_score += score
                if (j+1)%4==0:
                    outfile.write(str(sent_level_score)+"\n")
                    sent_level_score = 0
                else:
                    continue
