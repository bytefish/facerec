function Y = project(X, W, mu)
	%% Projects X onto W, pass mu to adjust mean.
	%%
	%% Args:
	%%  X: Array with observations given in columns.
	%%  W: Array representing the transformation matrix.
	%%  mu: Pass if the mean should be adjusted.
	%% 
	%%
	%% Returns:
	%%  Y: Projection of X.
	%%
	X = X - repmat(mu, 1, size(X,2));
	Y = W'*X;
endfunction
