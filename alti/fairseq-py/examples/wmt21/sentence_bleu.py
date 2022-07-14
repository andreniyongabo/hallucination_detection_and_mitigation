import sacrebleu
import pdb

ref = "/private/home/chau/wmt21/all_eval_sets/ha-en/newsdev2021.ha-en.en"

hyp1_scores = []
hyp1 = "/checkpoint/chau/wmt21/wmt_only_rev.bitext_bt.v3.64_shards/dense.fp16.mfp16.uf1.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.fully_sharded.det.mt8000.transformer.ELS24.DLS24.encffnx16384.decffnx16384.E2048.H32.NBF.ATTDRP0.1.RELDRP0.0.ngpu128/multi_benchmark/ha-en/newsdev2021.checkpoint_7_50000-shard0.wmtdata_newsdomain.lenpen1.0.beam4.hyp"
with open(ref) as ref_f, open(hyp1) as hyp_f:
    for ref_line, hyp_line in zip(ref_f, hyp_f):
        ref_str = ref_line.strip()
        hyp_str = hyp_line.strip()
        bleu = sacrebleu.sentence_bleu(hyp_str, [ref_str]).score
        hyp1_scores.append((bleu, hyp_str, ref_str))

hyp2_scores = []
hyp2 = "/checkpoint/chau/wmt21/bitext_bt/ha_en.bitext_bt.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.3.wd0.0.ls0.2.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.3.RELDRP0.3.ngpu32/multi_benchmark/ha-en/newsdev2021.checkpoint_best.wmtdata_newsdomain.lenpen1.0.beam4.hyp"
with open(ref) as ref_f, open(hyp2) as hyp_f:
    for ref_line, hyp_line in zip(ref_f, hyp_f):
        ref_str = ref_line.strip()
        hyp_str = hyp_line.strip()
        bleu = sacrebleu.sentence_bleu(hyp_str, [ref_str]).score
        hyp2_scores.append((bleu, hyp_str, ref_str))
diff = [(tup1[0] - tup2[0], tup1[0], tup2[0], tup1[1], tup2[1], tup1[2]) for tup1, tup2 in zip(hyp1_scores, hyp2_scores)]


sorted_diff = sorted(diff, key=lambda tup: tup[0])
for i in range(10):
    tup = sorted_diff[i]
    print(tup[1], tup[3])
    print(tup[2], tup[4])
    print(tup[5])
    print("-"*20)

print("="*20)
print('\n')
for i in range(10):
    tup = sorted_diff[-(i+1)]
    print(tup[1], tup[3])
    print(tup[2], tup[4])
    print(tup[5])
    print("-"*20)
