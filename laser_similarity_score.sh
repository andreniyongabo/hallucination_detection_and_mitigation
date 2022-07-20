#!bin/bash
"""
Getting similarity scores of the translation output of flores devtest
"""
export LASER=$HOME/LASER

echo "LASER path = ${LASER}"

DATADIR=/large_experiments/nllb/mmt/multilingual_bin/flores200.en_xx_en.v4.4.256k/retrieved_data
OUTPUTDIR=$HOME/hallucination_detection_and_mitigation/translations/flores_test
LASERDIR=$HOME/LASER

src="eng"
tgt="kin"

src_input_file=${DATADIR}/test.${src}-${tgt}.${src}
tgt_input_file=${OUTPUTDIR}/${src}-${tgt}/output.hyp
out_file=${OUTPUTDIR}/${src}-${tgt}/output.laser_score

# # uncomment this part to get the similarity scores of all beam candidates
# OUTPUTDIR=$HOME/hallucination_detection_and_mitigation/translations/flores_test/beam_candidates
# LASERDIR=$HOME/LASER

# src="eng"
# tgt="kin"

# beam=4 # remember to use the same beam size that were used when running "get_beam_candidates.sh"

# src_input_file=${OUTPUTDIR}/${src}-${tgt}/output_bms_${beam}_source.txt
# tgt_input_file=${OUTPUTDIR}/${src}-${tgt}/output_bms_${beam}_target.txt
# out_file=${OUTPUTDIR}/${src}-${tgt}/output_bms_${beam}_sim_scores.txt

mkdir ${OUTPUTDIR}/temp_embeddings
mkdir ${OUTPUTDIR}/temp_embeddings/${src}-${tgt}

if test -f "${OUTPUTDIR}/temp_embeddings/${src}-${tgt}/${src}_laser.embed"
then
    rm ${OUTPUTDIR}/temp_embeddings/${src}-${tgt}/${src}_laser.embed
    src_output_file=${OUTPUTDIR}/temp_embeddings/${src}-${tgt}/${src}_laser.embed
else
    src_output_file=${OUTPUTDIR}/temp_embeddings/${src}-${tgt}/${src}_laser.embed
fi
if test -f "${OUTPUTDIR}/temp_embeddings/${src}-${tgt}/${tgt}_laser.embed"
then
    rm ${OUTPUTDIR}/temp_embeddings/${src}-${tgt}/${tgt}_laser.embed
    tgt_output_file=${OUTPUTDIR}/temp_embeddings/${src}-${tgt}/${tgt}_laser.embed
else
    tgt_output_file=${OUTPUTDIR}/temp_embeddings/${src}-${tgt}/${tgt}_laser.embed
fi

bash ${LASERDIR}/tasks/embed/embed.sh ${src_input_file} ${src_output_file} # add the source lang token if it is not "eng"
bash ${LASERDIR}/tasks/embed/embed.sh ${tgt_input_file} ${tgt_output_file} ${tgt}

python laser_similarity_score.py ${src_output_file} ${tgt_output_file} ${out_file}

