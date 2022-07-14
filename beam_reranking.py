import numpy as np
import pandas as pd
import json
"""
Scripts to rerank the beam based on the sentence score and laser score
"""
def readJSON(json_file):
    with open(json_file, "r") as jfile:
        data = json.load(jfile)
    return data

def writeJSON(data, save_path):
    json_object = json.dumps(data, indent=4)
    with open(save_path, "w") as jfile:
        jfile.write(json_object)

def writeTXT(data, save_path):
    with open(save_path, "w") as tfile:
        for line in data:
            tfile.write(line+"\n")

def saveCSV(df, save_path):
    df.to_csv(save_path, index=False)
########
def txtToList(txt_file):
    out_list = []
    with open(txt_file, "r") as infile:
        for line in infile:
            out_list.append(float(line.strip()))
    return out_list

def addSimScores(json_file, sim_score_file, bms):
    data = readJSON(json_file)
    sim_scores_list = txtToList(sim_score_file)
    dict_idx = 0
    sim_scores = []
    for j in range(len(sim_scores_list)):
        sim_scores.append(sim_scores_list[j])
        if (j+1)%bms==0:
            data[str(dict_idx)]["sim_score"] = sim_scores
            dict_idx += 1
            sim_scores = []
        else:
            continue
    return data

def RerankedBeam(json_file, sim_scores_file, bms, order_by="sent_sim"):
    data = addSimScores(json_file, sim_scores_file, bms)
    new_data = {}
    for i, sample in enumerate(data):
        source_sents = data[str(i)]["source"]
        target_sents = data[str(i)]["target"]
        sent_scores = data[str(i)]["sent_score"]
        sim_scores = data[str(i)]["sim_score"]
        data[str(i)]["sent_sim"] = [sent_scores[j]+sim_scores[j] for j in range(len(sent_scores))]
        if order_by=="sent_sim":
            sorted_idx = np.argsort(-1*np.array(data[str(i)]["sent_sim"])) # multiply with -1 for descending sorting
        elif order_by=="sim_score":
            sorted_idx = np.argsort(-1*np.array(data[str(i)]["sim_score"])) # multiply with -1 for descending sorting
        else:
            print("Invalid input. Choose between 'sent_sim' and 'sim_score'.")
        data[str(i)]["target"] = [data[str(i)]["target"][si] for si in sorted_idx]
        data[str(i)]["sent_sim"] = [data[str(i)]["sent_sim"][si] for si in sorted_idx]
        data[str(i)]["sent_score"] = [data[str(i)]["sent_score"][si] for si in sorted_idx]
        data[str(i)]["sim_score"] = [data[str(i)]["sim_score"][si] for si in sorted_idx]
        new_data[i] = data[str(i)]

    return new_data

def SelectTopCand(json_file, sim_scores_file, bms, save_json_path, save_txt_path, save_csv_path, order_by="sim_score"):
    data = RerankedBeam(json_file, sim_scores_file, bms, order_by)
    source = []
    target = []
    sent_score = []
    sim_score = []
    sent_sim_score = []
    for i, sample in enumerate(data):
        source.append(data[i]["source"][0])
        target.append(data[i]["target"][0])
        sent_score.append(data[i]["sent_score"][0])
        sim_score.append(data[i]["sim_score"][0])
        sent_sim_score.append(data[i]["sent_sim"][0])

    writeJSON(data, save_json_path)
    writeTXT(target, save_txt_path)
    df = pd.DataFrame({"src":source, "tgt":target, "sent_score":sent_score, "sim_score":sim_score, "sent_sim_score":sent_sim_score})
    saveCSV(df, save_csv_path)

    return df

if __name__=="__main__":
    DATADIR = "/private/home/andreniyongabo/hallucination_detection_and_mitigation/translations/flores_test/beam_candidates/"
    src = "eng"
    tgt = "kin"
    beam_size = 4
    order_by = "sent_sim"

    json_file = f"{DATADIR}/{src}-{tgt}/beam_rerank/output_bs1_bms_{beam_size}.json"
    sim_scores_file = f"{DATADIR}/{src}-{tgt}/beam_rerank/output_bs1_bms_{beam_size}_sim_scores.txt"
    save_json_path = f"{DATADIR}/{src}-{tgt}/beam_rerank/output_bs1_bms_{beam_size}_reranked.json"
    save_txt_path = f"{DATADIR}/{src}-{tgt}/beam_rerank/output_bs1_bms_{beam_size}_target_reranked.txt"
    save_csv_path = f"{DATADIR}/{src}-{tgt}/beam_rerank/output_bs1_bms_{beam_size}_reranked.csv"
    
    SelectTopCand(json_file, sim_scores_file, beam_size, save_json_path, save_txt_path, save_csv_path, order_by=order_by)