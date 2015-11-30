% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function model = pca(X, num_components)
  %%  Performs a PCA on X and stores num_components principal components.
  %%
  %%  Args:
  %%    X [dim x num_data] Input 
  %%    y [1 x num_data] Classes
  %%    num_components [int] Number of components to use.
  %%
  %%  Out:
  %%    model [struct] learned model
  %%      .name [char] name of this model
  %%      .W [dim x num_components] components identified by PCA
  %%      .num_components [int] mumber of components used in this model.
  %%      .mu [dim x 1] sample mean of X
  %%
  %%  Example:
  %%    pca(X, y, struct('num_components',100))
  %%
  if(nargin < 2)
    num_components=size(X,2)-1;
  end
  % center data
  mu = mean(X,2);
  X = X - repmat(mu, 1, size(X,2));
  % svd on centered data == pca
  [E,D,V] = svd(X ,'econ');
  % build model
  model.name = 'pca';
  model.D = diag(D).^2;
  model.D = model.D(1:num_components);
  model.W = E(:,1:num_components);
  model.num_components = num_components;
  model.mu = mu;
end
