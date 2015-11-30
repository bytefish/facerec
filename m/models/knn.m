% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function c = knn(P, y, Q,  k)
    %%  k-nearest neighbor classification.
    %%
    %%  Args:
    %%      P [dim x num_data] reference vectors
    %%      Q [dim x 1] query vector
    %%      y [1 x num_data] classes corresponding to P. (y = {1,2,...,n})
    %%      k [int] nearest neighbors used in this prediction
    %%
    %%  Returns:
    %%      c [int] Class identified by the majority of k neighbors.
    %%
    %%  Example:
    %%      P=[1,21,20,2,4,30;
    %%         1,21,20,2,4,30]
    %%      y=[1,3,3,2,2,3]
    %%      Q=[1;1]
    %% 
    %%      knn(P,Q,y,1) % returns 1
    %%      knn(P,Q,y,3) % returns 2
    %%      knn(P,Q,y,6) % returns 3
    %%
    n = size(P,2);    
    % clip k
    if (nargin == 3)
        k=1;
    elseif (k>n)
        k=n;
    end

    Q = repmat(Q, 1, n);
    distances = sqrt(sum(power((P-Q),2),1));
    [distances, idx] = sort(distances);
    y = y(idx);
    y = y(1:k);
    h = histc(y,(1:max(y)));
    [v,c] = max(h);
end
