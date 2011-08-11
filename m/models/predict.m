function c = predict(model, Q, opts)
	Q = project(model, Q);
	c = knn(model.P, model.y, Q, opts.k);
endfunction
