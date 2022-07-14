import sys
import numpy as np
from numpy.linalg import norm


# src = "eng"
# tgt = "umb"
src_file = sys.argv[1] #source sentence embeddings file
tgt_file = sys.argv[2] #targert sentence embeddings file
out_file = sys.argv[3] #output file
dim = 1024

src_emb = np.fromfile(src_file, dtype=np.float32, count=-1)
# print(src_emb.shape)
src_emb.resize(src_emb.shape[0] // dim, dim)
# print(src_emb.shape, src_emb)

tgt_emb = np.fromfile(tgt_file, dtype=np.float32, count=-1)
# print(tgt_emb.shape)
tgt_emb.resize(tgt_emb.shape[0] // dim, dim)
# print(tgt_emb.shape, tgt_emb)

def cos_sim(A,B):
    similarity = np.sum(A*B, axis=1)/(norm(A, axis=1)*norm(B, axis=1))
    return similarity

cos_similarity = cos_sim(src_emb, tgt_emb)

print(cos_similarity)

with open(out_file, "w") as outfile:
    for score in cos_similarity:
        outfile.write(str(score)+"\n")
