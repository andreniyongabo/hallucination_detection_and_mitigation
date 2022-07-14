import json

def is_float(s):
    result = False
    if s.count(".") == 1 and s.replace(".", "").isdigit():
            result = True
    return result

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
            if is_float(line) or line.isdigit():
                tfile.write(str(line)+"\n")
            else:
                tfile.write(line+"\n")

def saveCSV(df, save_path):
    df.to_csv(save_path, index=False)

def txtToList(txt_file):
    out_list = []
    with open(txt_file, "r") as infile:
        for line in infile:
            if is_float(line):
                out_list.append(float(line.strip()))
            elif line.isdigit():
                out_list.append(int(line.strip()))
            else:
                out_list.append(line.strip())
    return out_list
