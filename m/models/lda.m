% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function model = lda(X, y, num_components)
  %%  Performs a Linear Discriminant Analysis and returns the 
  %%  num_components components sorted descending by their 
  %%  eigenvalue. 
  %%
  %%  num_components is bound to the number of classes, hence
  %%  num_components = min(c-1, num_components)
  %%
  %%  Args:
  %%    X [dim x num_data] input data
  %%    y [1 x num_data] classes
  %%    num_components [int] number of components to keep
  %%  
  %%  Returns:
  %%    model [struct] Represents the learned model.
  %%      .name [char] name of the model
  %%      .num_components [int] number of components in this model
  %%      .W [array] components identified by LDA
  %%
  dim = size(X,1);
  c = max(y); 
  
  if(nargin < 3)
    num_components = c - 1;
  end
  
  num_components = min(c-1,num_components);
  
  meanTotal = mean(X,2);
  
  Sw = zeros(dim, dim);
  Sb = zeros(dim, dim);
  for i=1:c
    Xi = X(:,find(y==i));
    meanClass = mean(Xi,2);
    % center data
    Xi = Xi - repmat(meanClass, 1, size(Xi,2));
    % calculate within-class scatter
    Sw = Sw + Xi*Xi';
    % calculate between-class scatter
    Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
  end

  % solve the eigenvalue problem
  [V, D] = eig(Sb,Sw);
  
  % sort eigenvectors descending by eigenvalue
  [D,idx] = sort(diag(D), 1, 'descend');
  
  V = V(:,idx);
  % build model
  model.name = 'lda';
  model.num_components = num_components;
  model.D = D(1:num_components);
  model.W = V(:,1:num_components);
end
