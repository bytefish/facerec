function X = reconstruct(W, Y, mu)
	%%	Reonstruct Y from X using W, pass mu to adjust mean.
	%%
	%%	Args:
	%%		Y [num_components x num_data] projection
	%%		W [dim x num_components] transformation matrix
	%%		mu [dim x 1] sample mean
	%%
	%%	Returns:
	%%		X [dim x num_data] reconstruct data
	%%
	X = W*Y +repmat (mu ,size (Y,2 ) , 1);
end