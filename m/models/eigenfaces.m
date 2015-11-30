% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function model = eigenfaces(X, y, num_components)
  %%  Performs a PCA on X and stores num_components principal components.
  %%
  %%  Args:
  %%    X [dim x num_data] input data 
  %%    y [1 x num_data] classes
  %%    num_components [int] components to keep
  %%
  %%  Out:
  %%    model [struct] learned model
  %%      .name [char] name
  %%      .mu [dim x 1] sample mean of X
  %%      .num_components [int] number of components in this model
  %%      .W [dim x num_components] components identified by PCA
  %%      .P [num_components x num_data] projection of X
  %%
  %%  Example:
  %%    m_eigenface = eigenfaces(X, y, 100)
  if(nargin < 3)
    num_components=size(X,2)-1;
  end
  % perform pca
  Pca = pca(X, num_components);
  % build model
  model.name = 'eigenfaces';
  model.D = Pca.D;
  model.W = Pca.W;
  model.num_components = num_components;
  model.mu = Pca.mu;
  % project data
  model.P = model.W'*(X - repmat(Pca.mu, 1, size(X,2)));
  % store classes
  model.y = y;
end
