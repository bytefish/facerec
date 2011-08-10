function C = eigenfaces_predict(model, Xtest, k)
	Q = model.W' * (Xtest - model.mu);
	C = knn(model.P, model.y, Q, k);
endfunction
