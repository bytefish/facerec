% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function validation_result = KFoldCV(X, y, k, fun_train, fun_predict, per_fold, print_debug)
  %%
	%% Perform a k-fold cross-validation
	%%
	%% There may be a much simpler approach to do a Stratified K-Fold Cross validation, you 
	%% probably want to look at & translate the scikit-learn approach to MATLAB from:
	%% https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cross_validation.py
	%%
	%% Args:
	%%  X [dim x num_data] Dataset a k-fold cross validation is performed on.
	%%	y	[1 x num_data] Classes corresponding to observations in X.
	%%  k [1 x 1] number of folds
	%%  fun_train [function handle] function to build a model (__must__ return a model)
	%%  fun_predict [function handle] function to get a prediction from a model.
	%%	per_fold [bool] if per fold, then results are given for each fold (default 1). 
	%%  print_debug [bool] print debug (default 0)
	%%
	%% Returns:
	%%  [tp, fp, tn, fn]: cross-validation result (per fold or accumulated)
	%% Example:
	%%  see example.m
	%%
	validation_result = [];

	% set default options
	if ~exist('print_debug')
		print_debug = 0;
	end
	
	if ~exist('per_fold')
		per_fold=0;
	end

	% shuffle array (is there a function for this?)
	[d idx] = sort(rand(1, size(X,2)));
	X = X(:,idx);
	y = y(idx);
	
	% holds the cross validation result
	tp=0; fp=0; tn=0; fn=0;	

  % find the unique classes (TODO make all this independent of any label order)	
	C = max(y); % means y must be {1,2,3,...,C}
  % find minimum and maximum number of samples per class
  nmin = +inf;
  nmax = -inf;
	for i = 1:C
		idx = find(y==i);
		ni = length(idx);
    nmin = min(nmin,ni);
    nmax = max(nmax,ni);
  end
  % build fold indices
  foldIndices = zeros(C, nmax);
  for i = 1:C
    idx = find(y==i);
		foldIndices(i, 1:numel(idx)) = idx;
	end

  % adjust k (means there less than k examples in a class)
	if(nmin<k)
		k=nmin;
	end
	
	% instances per fold
	foldSize = floor(nmin/k);
	
	% calculate fold indices for Testset A, Trainingset B
	for i = 0:(k-1)
		%
		% Works like this:
		% (1) class1|ABBBBBBBBB| (2) class1|BABBBBBBBB| (k) ...
		%	    class2|ABBBBBBBBB|     class2|BABBBBBBBB|
		%     classN|ABBBBBBBBB|     classN|BABBBBBBBB|
		%
		if(print_debug)
			fprintf(1,'Processing fold %d.\n', i);
			if isoctave()
				fflush(stdout);
			end
		end
	 	
		l = i*foldSize+1;
		h = (i+1)*foldSize;
		testIdx = foldIndices(:, l:h);
		trainIdx = foldIndices(:, [1:(l-1), (h+1):nmin]);
		
		% reshape to row vector again
		testIdx = reshape(testIdx, 1, numel(testIdx));
		trainIdx = reshape(trainIdx, 1, numel(trainIdx));
		
		% train a model
		model = fun_train(X(:,trainIdx), y(:,trainIdx));
		
		% test the model
		for idx=testIdx
			% evaluate model and return prediction structure
			prediction = fun_predict(model, X(:,idx));
			% if you want to count [tn, fn] please add your code here
			if(prediction == y(idx))
				tp = tp + 1;
			else
				fp = fp + 1;
			end
		end
		
		% if you want to log results on a per fold basis
		if(per_fold)
			validation_result = [validation_result; [tp, fp, tn, fn]];
			tp=0; fp=0; tn=0; fn=0;
		end
	end
	
	% or set the accumulated result
	if(~per_fold)
		validation_result = [tp, fp, tn, fn];
	end
end
