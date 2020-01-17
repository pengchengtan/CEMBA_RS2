#Personalized Pagerank to find crutial transcription factors

from collections import Counter
import numpy as np

def buildgraph(i):
    global cluster_dmr, motif_dmr, dmrgene, motif, gene_all, c, genedict, geneidx
    print(i)
    if  dmrgene[list(cluster_dmr.intersection(motif_dmr[motif[i]]))]!= np.array([], dtype=object):
        motifgene = np.concatenate(dmrgene[list(cluster_dmr.intersection(motif_dmr[motif[i]]))])
        
    else:
        motifgene = np.array([], dtype=object)
        
    count = Counter(motifgene)
    return [geneidx[motif[i]], np.array([count[x] for x in gene_all[:,-1]]) * mch[c, genedict[motif[i]]]]
    

def normgraph(A):
    ngene= len(A)
    print('Normalize')
    A = A + A.T
    A = A - np.diag(np.diag(A))
    A = A + np.diag(np.sum(A, axis=0) == 0)
    P = A / np.sum(A, axis=1)[:, None]
    return P

def pagerank(P, node_weight, alpha=0.85):
    print('Propagate')
    pr = np.ones(len(P)) / len(P)
    for i in range(300):
        pr_new = (1 - alpha) * node_weight + alpha * np.dot(pr, P)
        delta = np.linalg.norm(pr - pr_new)
        pr = pr_new.copy()
        print('Iter ', i, ' delta = ', delta)
        if delta < 1e-6:
            break
    return pr

