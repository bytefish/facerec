function c = predict(model, Q, opts)
	%% Perform k-NN on a given model with Reference vectors in P
	%%
	%% Args:
	%%   opts:
	%%     k: k in knn (see knn.m)
	%%
	if(~isfield(opts,"k"))
		opts.k = 1;
	endif
	Q = project(model, Q);
	c = knn(model.P, model.y, Q, opts.k);
endfunction
