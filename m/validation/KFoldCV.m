function validation_result = KFoldCV(X, y, k, fun_train, fun_predict, per_fold, print_debug)
	%% Perform a k-fold cross-validation
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
	if ~exist("print_debug")
		print_debug = 0;
	end
	
	if ~exist("per_fold")
		per_fold=0;
	endif

	% shuffle array (is there a function for this?)
	[d idx] = sort(rand(1, size(X,2)));
	X = X(:,idx);
	y = y(idx);
		
	tp=0; fp=0; tn=0; fn=0;	
	
	% create class to image array, looks like this:
	%
	% c1  = [ I1, I2, I3;
	% c2  =   I4, I5, I6; 
	% c3  =   I7, I8, I9 ] 
	
	C = max(y); % means y must be {1,2,3,...,C}
	foldIndices = [];
	n = +inf;
	for i = 1:C
		idx = find(y==i);
		ni = length(idx);
		n = min(n,ni);
		% Don't do this at home...
		if(ni > size(foldIndices,2))
			foldIndices = resize(foldIndices,size(foldIndices,1),ni);
		endif
		idx = resize(idx, 1, size(foldIndices,2));
		foldIndices = [foldIndices; idx];
	endfor

	% adjust k (less than k examples in one class)
	if(n<k)
		k=n;
	endif

	% instances per fold
	foldSize = floor(n/k);

	% calculate fold indices for Testset A, Trainingset B
	for i = 1:k
		%
		% Works like this:
		% (1) class1|ABBBBBBBBB| (2) class1|BABBBBBBBB| (k) ...
		%	    class2|ABBBBBBBBB|     class2|BABBBBBBBB|
		%     classN|ABBBBBBBBB|     classN|BABBBBBBBB|
		%
		if(print_debug)
			printf("Processing fold %d.\n", i);
			fflush(stdout);
		endif
	 	
		l = i*foldSize;
		h = (i+1)*foldSize-1;

		testIdx = foldIndices(:, l:h);
		trainIdx = foldIndices(:, [1:(l-1), (h+1):n]);
		
		% reshape to row vector again
		testIdx = reshape(testIdx, 1, numel(testIdx));
		trainIdx = reshape(trainIdx, 1, numel(trainIdx));
		
		% train a model
		model = fun_train(X(:,trainIdx), y(:,trainIdx));
		
		% log per fold
		if(per_fold)
			tp=0; fp=0; tn=0; fn=0;
		endif
		
		% test the model
		for idx=testIdx
			% evaluate model and return prediction structure
			prediction = fun_predict(model, X(:,idx));
			% if you want to count [tn, fn] please add your code here
			if(prediction == y(idx))
				tp = tp + 1;
			else
				fp = fp + 1;
			endif
		endfor
		
		if(per_fold)
			validation_result = [validation_result; [tp, fp, tn, fn]];
		endif
	endfor
	% or set the accumulated result
	if(~per_fold)
		validation_result = [tp, fp, tn, fn];
	endif	
endfunction
