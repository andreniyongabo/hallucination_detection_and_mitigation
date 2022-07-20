#!bin/bash

PREMODELDIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.4.256k
SPM=/private/home/costajussa/sentencepiece/build/src/
NORM_PUNCT=/private/home/vedanuj/workspace/fairseq-py/examples/nllb/modeling/preprocessing/moses/normalize-punctuation.perl

REFDIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.4.256k/retrieved_data # directory for reference translations
HYPDIR=$HOME/hallucination_detection_and_mitigation/translations/flores_test/beam_candidates #directory for hypotheses

src="eng"
tgt="kin"

ref_input_file=${REFDIR}/test.${src}-${tgt}.${tgt}
beam="4"

hyp_input_file=${HYPDIR}/${src}-${tgt}/output_bms_${beam}_target_reranked_by_sent_score.txt
rerank_by_sim_score_file=${HYPDIR}/${src}-${tgt}/output_bms_${beam}_target_reranked_by_sim_score.txt
rerank_by_sent_sim_file=${HYPDIR}/${src}-${tgt}/output_bms_${beam}_target_reranked_by_sent_sim.txt

# tokenize with sentence piece
cat ${ref_input_file} | perl ${NORM_PUNCT} -l ${lang} | ${SPM}/spm_encode --model=${PREMODELDIR}/vocab_bin/sentencepiece.source.256000.model --output_format=piece > ref.txt
cat ${hyp_input_file} | perl ${NORM_PUNCT} -l ${lang} | ${SPM}/spm_encode --model=${PREMODELDIR}/vocab_bin/sentencepiece.source.256000.model --output_format=piece > hyp_baseline.txt
cat ${rerank_by_sim_score_file} | perl ${NORM_PUNCT} -l ${lang} | ${SPM}/spm_encode --model=${PREMODELDIR}/vocab_bin/sentencepiece.source.256000.model --output_format=piece > hyp_rerank_by_sim_score.txt
cat ${rerank_by_sent_sim_file} | perl ${NORM_PUNCT} -l ${lang} | ${SPM}/spm_encode --model=${PREMODELDIR}/vocab_bin/sentencepiece.source.256000.model --output_format=piece > hyp_rerank_by_sent_sim.txt

wc ref.txt
wc hyp_baseline.txt
wc hyp_rerank_by_sim_score.txt
wc hyp_rerank_by_sent_sim.txt

echo "BLEU for beam ranked based on sentence scores"
sacrebleu ref.txt -i hyp_baseline.txt -m bleu -b -w 4 --force
echo "BLEU for beam ranked based on similarity scores"
sacrebleu ref.txt -i hyp_rerank_by_sim_score.txt -m bleu -b -w 4 --force
echo "BLEU for beam ranked based on sentence and similarity scores"
sacrebleu ref.txt -i hyp_rerank_by_sent_sim.txt -m bleu -b -w 4 --force
echo "***********************************************************"
