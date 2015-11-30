% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function model = fisherfaces(X, y, num_components)
  %%  Fisherfaces (see Python version for description)
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
  %%      .W [dim x num_components] components identified by LDA
  %%      .P [num_components x num_data] projection of X
  %%
  %%  Example:
  %%    m_fisherface = fisherface(X, y)
  
  N = size(X,2);
  c = max(y);
  
  % set the num_components
  if(nargin < 3)
    num_components=c-1;
  end
  
  num_components = min(c-1, num_components);
  
  % reduce dim(X) to (N-c) (see paper [BHK1997])
  Pca = pca(X, (N-c));
  Lda = lda(project(X, Pca.W, Pca.mu), y, num_components);
  
  % build model
  model.name = 'lda';
  model.mu = repmat(0, size(X,1), 1);
  model.D = Lda.D;
  model.W = Pca.W*Lda.W;
  model.P = model.W'*X;
  model.num_components = Lda.num_components;
  model.y = y;
end
