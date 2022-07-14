import sys

input_file = sys.argv[1]
out_file = sys.argv[2]

def listToStr(lst):
    str = ''
    for word in lst:
        str += word+" "
    return str

sents = []
with open(input_file, "r") as infile:
    for line in infile:
        temp = line.split()
        sent_first_part = listToStr(temp[0:int(len(temp)/4)])
        sent_second_part = listToStr(temp[int(len(temp)/4):2*int(len(temp)/4)])
        sent_third_part = listToStr(temp[2*int(len(temp)/4):3*int(len(temp)/4)])
        sent_fourth_part = listToStr(temp[3*int(len(temp)/4):len(temp)])
        
        sents.append(sent_first_part)
        sents.append(sent_second_part)
        sents.append(sent_third_part)
        sents.append(sent_fourth_part)

with open(out_file, "w") as outfile:
    for line in sents:
        outfile.write(line+"\n")