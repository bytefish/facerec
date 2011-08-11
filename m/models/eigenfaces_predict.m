function C = eigenfaces_predict(model, Xtest, k)
	%%	Predicts nearest neighbor for given Eigenfaces model.
	%%
	%%	Args:
	%%		model [struct] model for prediction
	%%		Xtest [dim x 1] test vector to predict
	Q = model.W' * (Xtest - model.mu);
	C = knn(model.P, model.y, Q, k);
endfunction
