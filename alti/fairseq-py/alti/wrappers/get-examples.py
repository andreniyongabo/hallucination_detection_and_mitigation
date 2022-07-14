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
    model = []

    for r,b,m in zip(args.refs, args.baseline, args.model):
        refs.append(read_file(r))
        base.append(read_file(b))
        model.append(read_file(m))

    return refs, base, model

    

def find_examples(refs,base,model,n=10):
    diff_score = []

    #compute BLEU score against the reference for each baseline and model output
    for i in range(len(refs[0])):
        b_score = sum([compute_bleu(refs[j][i],base[j][i]) for j in range(len(base))])/len(base)
        m_score = sum([compute_bleu(refs[j][i],model[j][i]) for j in range(len(model))])/len(model)
        diff_score.append((i,m_score-b_score))
        print('Computed score', i, m_score, b_score)


    #sort differences in descending order
    diff_score.sort(key=lambda x:x[1], reverse=True)

    return diff_score[:n]

def print_examples(refs,base,model,scores):

    for s in scores:
        idx, score = s
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('* EXAMPLE', idx, 'SCORE', score)
        print('---------------------------------------------------')
        print('* REFERENCES:')
        for i in range(len(refs)):
            print('  * REF',i,':', refs[i][idx])
        print('---------------------------------------------------')
        print('* BASELINES:')
        for i in range(len(base)):
            print('  * BAS',i,':', base[i][idx])
        print('---------------------------------------------------')
        print('* MODEL:')
        for i in range(len(model)):
            print('  * MOD',i,':', model[i][idx])
        print('---------------------------------------------------')
        print('---------------------------------------------------\n')



parser = argparse.ArgumentParser(description='Find examples in text according to sentence BLEU')
parser.add_argument('-r', '--refs', nargs='+' , help='Reference files to compute BLEU', required=True)
parser.add_argument('-b', '--baseline', nargs='+' , help='Baseline output files to compute BLEU', required=True)
parser.add_argument('-m', '--model', nargs='+' , help='Model output files to compute BLEU', required=True)
parser.add_argument('-n', '--numex', type=int , help='Number of examples required', default=10)
args = parser.parse_args()

assert len(args.refs) == len(args.baseline), "The number of references must be equal to the baseline outputs"
assert len(args.model) == len(args.baseline), "The number of model outputs must be equal to the baseline outputs"

refs,base,model = load_data(args)

scores = find_examples(refs,base,model,args.numex)

print_examples(refs,base,model,scores)

