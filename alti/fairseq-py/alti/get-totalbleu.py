import argparse
import sacrebleu


#Le pasas la referencia, un baseline, una  salida de tu modelo y cuantos ejemplos quieres
#Y te da las n frases con mayor diferencia de BLEU entre tu modelo y el baseline

def read_file(path):
    with open(path,'r') as f:
        return list(f.readlines())

def compute_bleu(reference, candidate):
    return sacrebleu.corpus_bleu([candidate],[[reference]]).score


def load_data(args):
    refs = []
    base = []


    for r,b in zip(args.refs, args.baseline):
        refs.append(read_file(r))
        base.append(read_file(b))


    return refs, base

    

    #compute BLEU score against the reference for each baseline and model output



parser = argparse.ArgumentParser(description='Find BLEU')
parser.add_argument('-r', '--refs', nargs='+' , help='Reference files to compute BLEU', required=True)
parser.add_argument('-b', '--baseline', nargs='+' , help='Baseline output files to compute BLEU', required=True)
args = parser.parse_args()

assert len(args.refs) == len(args.baseline), "The number of references must be equal to the baseline outputs"


refs,base = load_data(args)




for i in range(len(refs[0])):
    b_score = sum([compute_bleu(refs[j][i],base[j][i]) for j in range(len(base))])/len(base)
    print(b_score)
        


