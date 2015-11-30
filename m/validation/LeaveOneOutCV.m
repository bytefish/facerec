% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function validation_result = LeaveOneOutCV(X, y, fun_train, fun_predict, print_debug)
	%% Performs a Leave-One-Out-Cross validation.
	%%
	%% Args:
	%%  see KFoldCV.m
	
	if(~exist('print_debug'))
		print_debug = 0;
	end
	
	% shuffle dataset
	[d idx] = sort(rand(1, size(X,2)));
	X = X(:,idx);
	y = y(idx);
	
	tp = 0; fp = 0; tn = 0; fn = 0;
	n = length(y);
	for i = 1:n
		if(print_debug)
			fprintf(1,'Processing fold %d/%d.\n', i, n);
			if isoctave()
				fflush(stdout);
			end
		end
		
		Xi = X(:,1); X(:,1) = [];
		yi = y(1); y(1) = [];
	  
		model = fun_train(X, y);
		prediction = fun_predict(model, Xi);

		%% if you want to count [tn, fn] please add your code here
		if(prediction == yi)
			tp = tp + 1;
		else
			fp = fp + 1;
		end
		
		% add to test instance end of list
		X = [X, Xi];
		y = [y, yi];
	end
		
	validation_result = [tp fp tn fn];
	
end
