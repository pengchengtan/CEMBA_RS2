#Calculate factors explained variance
#data: sample data
#meta: metadata for each sample
#factors: class name for factor to be calculated, contained in metadata. e.g. ['male', 'female'] for gender as factor

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def Explained_variance(data, meta, factors):
  #construct factor vector
  vector = np.full((len(meta),len(factors),0)
  for i in range(len(meta)):
      for k in range(len(factors)):
          if meta[i] == factors[k]:
              vector[i][k] = 1
  #calculate vector PCA reduction
  pca = PCA(n_components=len(factors))
  vector_reduce = pca.fit_transform(vector)
  #calculate variance
  model = LinearRegression().fit(data, vector_reduce)
  xvector = normalize(model.coef_.T, axis = 0)
  varvector = np.sum(np.var(np.dot(data,xvector), axis = 0))
  #calculate data variance
  pca = PCA(n_components=len(factors))
  data_reduce = pca.fit_transform(data)
  variance = np.sum(pca.explained_variance_)
  #normalize factor variance
  explained_ratio = varvactor/variance
  return explained_ratio

