% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

% load function files from subfolders aswell
addpath (genpath ('.'));
% load data
[X y width height names] = read_images('/home/philipp/facerec/data/yalefaces_recognition');
% number of samples
n = size(X,2);
% get a random index
randomIdx = uint32(rand()*n);
% split data
% into training set
Xtrain = X(:, [1:(randomIdx-1), (randomIdx+1):n]); 
ytrain = y([1:(randomIdx-1), (randomIdx+1):n]);
% into test set
Xtest = X(:,randomIdx);
ytest = y(randomIdx);
% compute a model
model = fisherfaces(Xtrain,ytrain);
% get a prediction from the model
predicted = fisherfaces_predict(model, Xtest, 1);
% only for debug
fprintf(1,'predicted=%d,actual=%d\n', predicted, ytest)
