#Visualization by t-SNE or UMAP

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE
import umap

def run_tSNE(data, n_pc, n_dim, p, verbose = 3, random_state = 0, n_jobs = 20):
  pca = PCA(n_components = n_pc)
  rateb_reduce = pca.fit_transform(data)
  ndim = n_dim
  tsne = MulticoreTSNE(perplexity = p, verbose = verbose, random_state = random_state, n_jobs = n_jobs)
  y = tsne.fit_transform(rateb_reduce[:, :n_dim])
  return y

def run_UMAP(data, n_pc, n_dim, n_neighbors = 15, random_state = 0):
  pca = PCA(n_components = n_pc)
  rateb_reduce = pca.fit_transform(data)
  ndim = n_dim
  Umap = umap.UMAP(n_neighbors = n_neighbors, random_state = random_state)
  y = Umap.fit_transform(rateb_reduce[:, :n_dim])
  return y

def draw_tSNE(path, y, color, cmap, cbarlabel, vmin, vmax):
  fig, ax = plt.subplots(figsize=(8,6))
  ax.set_xlabel('t-SNE-1', fontsize =20)
  ax.set_ylabel('t-SNE-2', fontsize =20)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.tick_params(axis='both', which='both', length=0)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  plot = ax.scatter(y[:, 0], y[:, 1], s=3, c=color, alpha=0.8, edgecolors='none', cmap=cmap, rasterized = True)
  cbar = plt.colorbar(plot, ax=ax)
  cbar.solids.set_clim([vmin, vmax])
  cbar.set_ticks([vmin, vmax])
  cbar.ax.tick_params(labelsize = 16)
  cbar.set_label(cbarlabel)
  cbar.draw_all()
  plt.savefig(path, transparent=True, bbox_inches='tight', dpi=300)
  plt.close()
  return plt


def draw_UMAP(path, y, color, cmap, cbarlabel, vmin, vmax):
  fig, ax = plt.subplots(figsize=(8,6))
  ax.set_xlabel('UMAP-1', fontsize =20)
  ax.set_ylabel('UMAP-2', fontsize =20)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.tick_params(axis='both', which='both', length=0)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  plot = ax.scatter(y[:, 0], y[:, 1], s=3, c=color, alpha=0.8, edgecolors='none', cmap=cmap, rasterized = True)
  cbar = plt.colorbar(plot, ax=ax)
  cbar.solids.set_clim([vmin, vmax])
  cbar.set_ticks([vmin, vmax])
  cbar.ax.tick_params(labelsize = 16)
  cbar.set_label(cbarlabel)
  cbar.draw_all()
  plt.savefig(path, transparent=True, bbox_inches='tight', dpi=300)
  plt.close()
  return plt


