#read and store homer motif enrichment results

import numpy as np

nc = num_cluster
pv = []
tarc = []
basec = []

indir = ' '
for i in range(nc):
    data = np.loadtxt(indir + 'cluster'+str(i)+'/knownResults.txt', dtype=np.str, delimiter='\t', skiprows=1)
    index = np.argsort(data[:,0])
    pv.append(data[index, 3])
    tarc.append(data[index, 6])
    basec.append(data[index,8])

pv = np.array(pv)
tarc = np.array(tarc)
basec = np.array(basec)
data[index,0] = np.array(data[index,0])
pvall = np.concatenate((data[index, 0][:,None], pv.T),axis = 1)
tarcall = np.concatenate((data[index, 0][:,None], tarc.T),axis = 1)
basecall = np.concatenate((data[index, 0][:,None], basec.T),axis = 1)
fc = np.arange(len(tarc)*len(tarc[0])).reshape((len(tarc),len(tarc[0])))
for i in range(len(tarc)):
    for j in range(len(tarc[i])):
        tarc[i][j] = tarc[i][j][:-1]
        basec[i][j] = basec[i][j][:-1]

tarc = tarc.astype(float)
basec = basec.astype(float)
pv = pv.astype(float)
fc = (tarc-basec)/basec
fcall = np.concatenate((data[index, 0][:,None], fc.T),axis = 1)
