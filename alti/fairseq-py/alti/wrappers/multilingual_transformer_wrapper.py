import logging
import warnings
from functools import partial
from collections import defaultdict

# From hub.utils
import copy
from typing import Any, Dict, Iterator, List
from omegaconf import open_dict
from fairseq import utils
logger = logging.getLogger(__name__)


MODEL_LANGS="afr,amh,ara,asm,ast,aym,azj,bel,ben,bos,bul,cat,ceb,ces,ckb,cym,dan,deu,dyu,ell,eng,est,fas,fin,fra,ful,gla,gle,glg,guj,hat,hau,heb,hin,hrv,hun,hye,ibo,ilo,ind,isl,ita,jav,jpn,kac,kam,kan,kat,kaz,kea,khm,kir,kmb,kon,kor,kur,lao,lav,lin,lit,ltz,lug,luo,mal,mar,mkd,mlg,mlt,mon,mri,msa,mya,nld,nob,npi,nso,nya,oci,orm,ory,pan,pol,por,pus,que,ron,rus,sin,slk,slv,sna,snd,som,spa,sqi,srp,ssw,sun,swe,swh,tam,tel,tgk,tgl,tha,tir,tsn,tur,ukr,umb,urd,uzb,vie,wol,xho,yid,yor,yue,zho_Hans,zul".split(",")

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.transformer_wrapper import FairseqTransformerHub
from fairseq.models.transformer import TransformerModel

from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns


class FairseqMultilingualTransformerHub(FairseqTransformerHub):

    def __init__(self, cfg, task, models):
        super().__init__(cfg, task, models)
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")
    

    @classmethod
    def from_pretrained(cls, checkpoint_dir, checkpoint_file, data_name_or_path, dict_path, source_lang, target_lang, lang_pairs):#, sentencepiece_model):#added sentencepiecemodel
        hub_interface = TransformerModel.from_pretrained(checkpoint_dir, checkpoint_file, data_name_or_path,
                                                        source_lang=source_lang,
                                                        target_lang=target_lang,
                                                        add_lang_tok=False,
                                                        lang_pairs=lang_pairs,
                                                        fixed_dictionary =dict_path,
                                                        langs=MODEL_LANGS, 
                                                        task="translation_multi_simple_epoch",#added
                                                        encoder_langtok="tgt",#added
                                                        decoder_langtok=True,#added
                                                        #skip-invalid-size-inputs-valid-test="",#added
                                                        beam=4,#added
                                                        max_tokens=512,#added
                                                        bpe="sentencepiece",#added
                                                        sentencepiece_model="/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/vocab_bin/sentencepiece.source.256000.model",#added
                                                        #fixed_dictionary =f'{data_name_or_path}/dict.eng.txt'
                                                        )

# ./jigsaw-corpus/ \
#        --path /large_experiments/nllb/mmt/h2_21_models/flores125_v3.3/en_to_many_to_en/v3.3_dense_hrft004.mfp16.mu100000.uf4.lss.enttgt.tmp1.0.shem.NBF.\warmup8000.lr0.004.drop0.0.maxtok2560.seed2.valevery200000.max_pos512.adam16bit.fully_sharded.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H\16.ATTDRP0.1.RELDRP0.0.ngpu128/checkpoint_15_100000_consolidated.pt \
#        --task=translation_multi_simple_epoch \
#        --langs ${MODEL_LANGS} \
#        --lang-pairs eng-$lang \
#        --source-lang eng --target-lang $lang \
#        --encoder-langtok "tgt" --decoder-langtok \
#        --gen-subset test \
#        --skip-invalid-size-inputs-valid-test \
#        --beam 4 \
#        --max-tokens 512 \
#        --bpe "sentencepiece" \
#        --sentencepiece-model ${DATADIR}/vocab_bin/sentencepiece.source.256000.model > ./jigsaw-translations/translation-all.$lang

        #print('classfrompretrained',len(hub_interface.task.target_dictionary)) 
        return cls(hub_interface.cfg, hub_interface.task, hub_interface.models)

    def decode(self, tensor, dictionary, as_string=False):
        tok = []
        for token in torch.squeeze(tensor):
                tok.append(dictionary[token])
        # tok = dictionary.string(tensor,'sentencepiece').split()
        if as_string:
            return ''.join(tok).replace('▁', ' ')
        else:
            return tok

    def get_sample(self, split, index):

        if split not in self.task.datasets.keys():
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]['source']
        src_tok = self.decode(src_tensor, self.task.source_dictionary)
        src_sent = self.decode(src_tensor, self.task.source_dictionary, as_string=True)

        # Adding EOS token to beggining of target sentence
        tgt_tensor = self.task.dataset(split)[index]['target']
        #tgt_tensor = torch.cat([torch.tensor([self.task.target_dictionary.eos_index]),
        #                        tgt_tensor[:-1]
        #                        ]).to(tgt_tensor.device)
        tgt_tok = self.decode(tgt_tensor, self.task.target_dictionary)
        tgt_sent = self.decode(tgt_tensor, self.task.target_dictionary, as_string=True)

        return src_sent, src_tok, src_tensor, tgt_sent, tgt_tok, tgt_tensor

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
        #self.zero_grad()
        with torch.no_grad():

            layer_inputs = defaultdict(list)
            layer_outputs = defaultdict(list)

            def save_activation(name, mod, inp, out):
                layer_inputs[name].append(inp)
                layer_outputs[name].append(out)

            handles = {}

            for name, layer in self.named_modules():
                handles[name] = layer.register_forward_hook(partial(save_activation, name))
            
            src_tensor = src_tensor.unsqueeze(0).to(self.device)

            # tgt_tensor = torch.cat([
            #     torch.tensor([self.task.target_dictionary.eos_index]),
            #     tgt_tensor[:-1]
            # ]).unsqueeze(0).to(self.device)
            tgt_tensor = tgt_tensor.unsqueeze(0).to(self.device)

            model_output, encoder_out = self.models[0](src_tensor, src_tensor.size(-1), tgt_tensor, )

            log_probs = self.models[0].get_normalized_probs(model_output, log_probs=True, sample=None)
            
            for k, v in handles.items():
                handles[k].remove()
            
            return model_output, log_probs, encoder_out, layer_inputs, layer_outputs

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        #inference_step_args: int = 0,
        skip_invalid_size_inputs=False,
        inference_step_args=None, #None
        prefix_allowed_tokens_fn=None,
        **kwargs
        ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(
            self.models,
            gen_args,
            #prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        inference_step_args = inference_step_args or {} 
        #inference_step_args={"rank": 0}#{} #added 0
        results = []
                
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs, rank=0,world_size=1,batch_size=1):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
#            print (batch['src_tokens'].shape)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo["tokens"],self.task.target_dictionary)
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    logger.info(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    if hypo["alignment"] is not None and getarg(
                        "print_alignment", False
                    ):
                        logger.info(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )
        return outputs