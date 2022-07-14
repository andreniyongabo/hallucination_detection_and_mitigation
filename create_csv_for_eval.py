import os
import pandas as pd

src = "eng"
tgts = ['wol', 'ibo', 'kam', 'luo', 'orm', 'som', 'yor']
# ['wol', 'ibo', 'kam', 'luo', 'orm', 'som', 'yor']
#, 'hau', 'lin', 'lug', 'nya', 'swh', 'tsn', 'umb'
for tgt in tgts:
    src_file = f"/private/home/costajussa/interpretability/nmt/humaneval/translation.catastrophic.{tgt}-{src}.{src}"
    tgt_file = f"/private/home/costajussa/interpretability/nmt/humaneval/catastrophic.{tgt}-{src}.{tgt}"
    non_cat_src_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/non-catastrophic-{tgt}-{src}.{tgt}"
    non_cat_tgt_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/non-catastrophic-{tgt}-{src}.{src}"
    # tgt_nllb200_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/output_bs1_bms4.hyp"
    sent_score_orig_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/output_bs1_bms4_orig.score"
    non_cat_sent_score_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/non-catastrophic-{tgt}-{src}-sent-score.txt"
    # sent_score_nllb200_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/output_bs1_bms4.score"
    sim_score_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/{tgt}-{src}_laser_sim.score"
    non_cat_sim_score_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/non-catastrophic-{tgt}-{src}_laser_sim_orig.score"
    # sim_score_nllb200_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/{tgt}-{src}_laser_sim_nllb200.score"
    alti_score_orig_file = f"/private/home/andreniyongabo/mytest/alti_inter/alti/data/{tgt}-{src}_src_contributions_untranslated.txt"
    non_cat_alti_score_file = f"/private/home/andreniyongabo/mytest/translations/catastrophic/{tgt}-{src}/non-catastrophic-{tgt}-{src}-alti-score.txt"
    out_file = "/private/home/andreniyongabo/mytest/evaluation/catastrophic"

    # src_file = f"/private/home/andreniyongabo/mytest/data/flores_test/test.{src}"
    # tgt_file = f"/private/home/andreniyongabo/mytest/translations/flores_test/{src}-{tgt}/output_bs1_bms4.hyp"
    # sent_score_orig_file = f"/private/home/andreniyongabo/mytest/translations/flores_test/{src}-{tgt}/bs1_bms4_translation_scores.txt"
    # sim_score_file = f"/private/home/andreniyongabo/mytest/translations/flores_test/{src}-{tgt}/output_bs1_bms_4_sim_scores.txt"
    # out_file = "/private/home/andreniyongabo/mytest/evaluation/flores_test"

    src_sent = []
    non_cat_src_sent = []
    tgt_sent = []
    non_cat_tgt_sent = []
    # tgt_nllb200_sent = []
    sent_score_orig = []
    non_cat_sent_score = []
    # sent_score_nllb200 = []
    sim_score = []
    non_cat_sim_score = []
    # sim_score_nllb200 = []
    alti_score_orig = []
    non_cat_alti_score = []

    with open(src_file, "r") as fp:
        for line in fp:
            src_sent.append(line)
    with open(non_cat_src_file, "r") as fp:
        for line in fp:
            non_cat_src_sent.append(line)
    with open(tgt_file, "r") as fp:
        for line in fp:
            tgt_sent.append(line)
    with open(non_cat_tgt_file, "r") as fp:
        for line in fp:
            non_cat_tgt_sent.append(line)
    # with open(tgt_nllb200_file, "r") as fp:
    #     for line in fp:
    #         tgt_nllb200_sent.append(line)
    with open(sent_score_orig_file, "r") as fp:
        for line in fp:
            sent_score_orig.append(line)
    with open(non_cat_sent_score_file, "r") as fp:
        for line in fp:
            non_cat_sent_score.append(line)
    # with open(sent_score_nllb200_file, "r") as fp:
    #     for line in fp:
    #         sent_score_nllb200.append(line)
    with open(sim_score_file, "r") as fp:
        for line in fp:
            sim_score.append(line)
    with open(non_cat_sim_score_file, "r") as fp:
        for line in fp:
            non_cat_sim_score.append(line)
    # with open(sim_score_nllb200_file, "r") as fp:
    #     for line in fp:
    #         sim_score_nllb200.append(line)
    with open(alti_score_orig_file, "r") as fp:
        for line in fp:
            alti_score_orig.append(line)
    with open(non_cat_alti_score_file, "r") as fp:
        for line in fp:
            non_cat_alti_score.append(line)

    df_alti = pd.read_csv(f"/private/home/andreniyongabo/mytest/evaluation/catastrophic/{tgt}-{src}-orig.csv")
    alti_score = list(df_alti["alti_score_orig"])

    new_src_sent = src_sent + non_cat_src_sent
    new_tgt_sent = tgt_sent + non_cat_tgt_sent
    new_sent_score = sent_score_orig + non_cat_sent_score
    new_sim_score = sim_score + non_cat_sim_score
    new_alti_score = alti_score + non_cat_alti_score
    evaluation = ["catastrophic"]*len(src_sent) + ["non_catastrophic"]*len(non_cat_src_sent)
    print(len(new_src_sent), len(new_tgt_sent), len(new_sent_score), len(new_sim_score), len(new_alti_score), len(evaluation))

    # df = pd.DataFrame({tgt:tgt_sent, "eng_orig":src_sent, "eng_nllb200":tgt_nllb200_sent, "sent_score_orig":sent_score_orig, "sent_score_nllb200":sent_score_nllb200, "sim_score_orig":sim_score, "sim_score_nllb200":sim_score_nllb200})
    df = pd.DataFrame({src:new_src_sent, tgt:new_tgt_sent, "sent_score":new_sent_score, "sim_score":new_sim_score, "alti_score":new_alti_score, "evaluation":evaluation})
    df.to_csv(f"{out_file}/{tgt}-{src}-balanced-eval.csv", index=False)
