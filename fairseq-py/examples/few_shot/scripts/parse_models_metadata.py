

# 1. Copy the table from https://docs.google.com/spreadsheets/d/1Mnqc3PpARkka-ETmwslM4xSf2EwB0Fc-yW-lFQ4z6_U/edit#gid=109744040
# 2. Run this script

if __name__ == "__main__":
    model_mapping = """
    Config names	Model	# params (B)	layers	hidden	seq len	train tokens	# experts	Extra MoE FLOPS per update	TFLOPS to train	V100 TFLOPS	V100 GPU days to train	Notes
    125M_gpt3_setting	GPT-3 125M	0.125	12	768	2048	300.0 B	0	0	363560141	30	140	
    openai_ada, 355M_gpt3_setting	GPT-3 355M	0.355	24	1024	2048	300.0 B	0	0	1058527642	30	408	
        GPT-3 760M	0.76	24	1536	2048	300.0 B	0	0	2131373261	30	822	
    1.3B_gpt3_setting	GPT-3 1.3B	1.3	24	2048	2048	300.0 B	0	0	3566606746	30	1376	
    openai_babbage, 2.7B_gpt3_setting	GPT-3 2.7B	2.7	32	2560	2048	300.0 B	0	0	7075504128	30	2730	
    openai_curie	GPT-3 6.7B	6.7	32	4096	2048	300.0 B	0	0	17119012454	30	6605	
    6.7B_gpt3_setting_1024ctx	Our 6.7B	6.7	32	4096	1024	300.0 B	0	0	16474767360	30	6356	
        GPT-3 13B	13	40	5120	2048	300.0 B	0	0	32673054720	30	12605	
    openai_davinci	GPT-3 175B	175	96	12288	2048	300.0 B	0	0	430173152870	30	165962	
        MoE 52B	52	12	2048	1024	75.5 B	256	121597190	563173704	30	217	Doesn't quite match 355M dense
    moe_52B	MoE 52B	52	24	1024	2048	300.0 B	512	241591661	1300118212	30	502	Matches 355M dense
    moe_207B	MoE 207B	207	24	2048	2048	300.0 B	512	966366645	4532969714	30	1749	Matches 1.3B dense
        MoE 264B	264	40	1792	1024	90.2 B	512	370667360	1637053843	30	632	Only partially trained
        MoE 523B	523	40	1792	1024	302.0 B	1024	1241304647	5482226824	30	2115	Doesn't quite match 2.7B dense
    moe_523B	MoE 523B	523	24	2304	1024	302.0 B	1024	1231171548	5407015280	30	2086	Doesn't match any dense model; has more experts
    moe_1.1T	MoE 1.1T	1104	32	4096	1024	300.0 B	512	5153883385	21628403428	30	8344	Matches 6.7B dense
    moe_15B	MoE 15B	15	12	768	2048	300.0 B	512	67947725	431507866	30	166	Matches 125M dense
    """

    import csv

    with open("models_params.tsv", mode="w") as f_out:
        f_out.write(model_mapping.strip())

    dict_items = {}
    with open("models_params.tsv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for item in reader:
            for k in list(item.keys()):
                try:
                    item[k] = int(item[k])                
                except:
                    try:
                        item[k] = float(item[k])
                    except:
                        pass
                
            for model in [x.strip() for x in item["Config names"].split(",")]:
                dict_items[model] = item


    print("{")
    for k, v in dict_items.items():
        print(f"\t'{k}': {v},")
    print("}")