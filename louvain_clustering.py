g = knn(rateb_reduce[:, :50],n_neighbors=25)
inter = g.dot(g.T)
diag = inter.diagonal()
jac = inter.astype(float)/(diag[None,:]+diag[:,None]-inter)
adj = nx.from_numpy_matrix(g.multiply(jac).toarray())
knnjaccluster = {}

for res in [0.8, 1.0, 1.2, 1.6, 2.0, 2.5]:
    partition = community.best_partition(adj,resolution=res)
    label = np.array([k for k in partition.values()])
    knnjaccluster[res] = label
    nc = len(set(label))
    count = np.array([sum(label==i) for i in range(nc)])
    print(res,count)
    fig, ax = plt.subplots()
    ax.set_xlabel('TSNE-1', fontsize=20)
    ax.set_ylabel('TSNE-2', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for i in range(max(label)+1):
        cell = (label==i)
        ax.scatter(y[cell,0], y[cell,1], s=5, c=color[i], alpha=0.8, edgecolors='none')
        ax.text(np.median(y[cell,0]), np.median(y[cell,1]), str(i), fontsize=12, horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    plt.savefig('/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/plot/cell_12536_posterior_disp2k_pc100_nd50_p50.pc50.knn25.louvain.res'+str(res)+ '_nc.'+str(nc)+'.pdf', bbox_inches="tight")
    plt.close()

    
np.save('/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/matrix/cell_12536_knnjaccluster_knn25.npy', knnjaccluster)

