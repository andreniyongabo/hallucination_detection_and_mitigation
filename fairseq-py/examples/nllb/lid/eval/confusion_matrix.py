#!python3 -u

import fasttext
import numpy as np
from matplotlib import pyplot as plt


def main():
    model_path = "/large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/result/model.bin"
    valid_path = "/large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/data/test/flores-dev.txt"

    # /private/home/celebio/nlp/nllb_lid/fastText/fasttext predict-prob /large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/result/model.bin /large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/data/test/flores-dev.txt | head -n 30
    print("Loading model")
    ft = fasttext.load_model(model_path)
    print("Loaded model")

    nb_langs = len(ft.labels)

    confusion = np.zeros((nb_langs, nb_langs))
    lang_index_map = {ft.labels[i] : i for i in range(nb_langs)}
    values = []
    ctr = 0
    with open(valid_path, "r") as fl:
        for line in fl:
            if line[-1] == "\n":
                line = line[:-1]

            # print(f"line = {line}")
            pred_labels, pred_probs = ft.predict(line, k=1, threshold=0.0, on_unicode_error='strict')
            pred_label = pred_labels[0]
            gold_label = line.split()[0]

            if gold_label in lang_index_map:
                pred_label_ind = lang_index_map[pred_label]
                gold_label_ind = lang_index_map[gold_label]

                confusion[pred_label_ind][gold_label_ind] = confusion[pred_label_ind][gold_label_ind]+1

            if ctr % 10000 == 0:
                print(f"line no = {ctr}")
            ctr+=1
            # if ctr > 10:
            #     break

            # ft.predict()
            # if ft_label_score_line.match(line):
            #     spl = line.split()
            #     values.append([float(spl[2]), float(spl[5]), float(spl[8]), spl[9][9:]])

    plt.imshow(confusion)
    plt.savefig("confusion.png", dpi=300)

    # From:
    # https://stackoverflow.com/a/50671617/411264
    # (but transposed (our `confusion`.T = their `cnf_matrix`))
    # ----------------------------------------------
    FP = confusion.sum(axis=1) - np.diag(confusion)
    FN = confusion.sum(axis=0) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)


    FPR[lang_index_map['__label__hrv']]
    FPR[lang_index_map['__label__eng']]
    FPR[lang_index_map['__label__ita']]
    FPR[lang_index_map['__label__ind']]


    inds = np.argsort(np.diagonal(confusion))

    confusion_order = confusion[:,inds][inds,:]

    labels_ordered = np.array(ft.labels)[inds]

    print(f"labels_ordered = {labels_ordered}")

    plt.imshow(confusion_order)
    plt.savefig("confusion_order.png", dpi=300)


    for column in list(range(20)) + [67]:
        print(f"Label = {labels_ordered[column]}")
        inds2 = np.argsort(-confusion_order[:,column])
        golds = confusion_order[:,column][inds2].sum()
        for row_no in range(5):
            nb_predicted = confusion_order[:,column][inds2][row_no]
            nb_predicted_perc = (nb_predicted * 1.0) / golds
            name_predicted = labels_ordered[inds2[row_no]]
            print(f"\tpredicted as {name_predicted} : {nb_predicted_perc:.3f}")
    # np.diagonal(conf)[inds]
    # np.diagonal(confusion)
    print("hello2")


if __name__ == '__main__':
    main()