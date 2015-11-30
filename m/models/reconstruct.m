% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function X = reconstruct(W, Y, mu)
    %%  Reonstruct Y from X using W, pass mu to adjust mean.
    %%
    %%  Args:
    %%    Y [num_components x num_data] projection
    %%    W [dim x num_components] transformation matrix
    %%    mu [dim x 1] sample mean (optional)
    %%
    %%  Returns:
    %%    X [dim x num_data] reconstruct data
    %%
    if(nargin<3)
        X = W * Y;
    else
        X = W*Y +repmat(mu,size(Y,2),1);
    end
end
