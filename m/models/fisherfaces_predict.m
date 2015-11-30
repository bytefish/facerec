% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function C = fisherfaces_predict(model, Xtest, k)
	%%	Predicts nearest neighbor for given Fisherfaces model.
	%%
	%%	Args:
	%%		model [struct] model for prediction
	%%		Xtest [dim x 1] test vector to predict
	Q = model.W' * Xtest;
	C = knn(model.P, model.y, Q, k);
end
