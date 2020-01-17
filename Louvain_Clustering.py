#Louvain clustering

import numpy as np
from sklearn.decomposition import PCA
import networkx as nx
from sklearn.neighbors import kneighbors_graph as knn
import matplotlib.pyplot as plt


def calculate_rateb_reduce(read_bin, rate_bin, binfilter, n_hvf, n_pc):
    mc = (read_bin * rate_bin)[:, binfilter]
    tc = read_bin[:, binfilter]
    rateb = calculate_posterior_mc_rate(mc, tc)
    disp = highly_variable_methylation_feature(rateb, np.mean(tc, axis=0), bins)
    idx = np.argsort(disp)[::-1]
    data = rateb[:, idx[:n_hvf]]
    pca = PCA(n_components=n_pc)
    rateb_reduce = pca.fit_transform(data)
    return rateb_reduce


g = knn(rateb_reduce[:, :ndim],n_neighbors=nn)
inter = g.dot(g.T)
diag = inter.diagonal()
jac = inter.astype(float)/(diag[None,:]+diag[:,None]-inter)
adj = nx.from_numpy_matrix(g.multiply(jac).toarray())
knnjaccluster = {}

list_of_res = [ ]
for res in list_of_res:
    partition = community.best_partition(adj,resolution=res)
    label = np.array([k for k in partition.values()])
    knnjaccluster[res] = label
    nc = len(set(label))
    count = np.array([sum(label==i) for i in range(nc)])
    print(res,count)
    fig, ax = plt.subplots()
    ax.set_xlabel('t-SNE-1', fontsize=20)
    ax.set_ylabel('t-SNE-2', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for i in range(max(label)+1):
        cell = (label==i)
        ax.scatter(y[cell,0], y[cell,1], s=5, c=color[i], alpha=0.8, edgecolors='none', rasterized=True)
        ax.text(np.median(y[cell,0]), np.median(y[cell,1]), str(i), fontsize=12, horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    plt.savefig(' ', bbox_inches='tight', dpi=300)
    plt.close()

    
np.save(' ', knnjaccluster)

