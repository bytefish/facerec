% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function Y = project(X, W, mu)
	%%	Projects X onto W, pass mu to adjust mean.
	%%
	%%	Args:
	%%		X [dim x num_data] input data
	%%		W [dim x num_components] transformation matrix
	%%		mu [dim x 1] sample mean
	%%
	%%	Returns:
	%%		Y [num_components x num_data] projection
	%%
	X = X - repmat(mu, 1, size(X,2));
	Y = W'*X;
end
