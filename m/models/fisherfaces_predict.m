function C = fisherfaces_predict(model, Xtest, k)
	Q = model.W' * Xtest;
	C = knn(model.P, model.y, Q, k);
endfunction
