% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function validation_result = LeaveOneClassOutCV(X, y, g, fun_train, fun_predict, print_debug)
	%% Performs a Leave-One-Out-Cross validation.
	%%
	%% Args:
	%%  g [1 x num_data] groups corresponding to classes of y
	%%  see KFoldCV.m
	
	if(~exist('print_debug'))
		print_debug = 0;
	end
	
	% shuffle idx
	[d idx] = sort(rand(1, size(X,2)));
	% shuffle X,y,g accordingly
	X = X(:,idx);
	y = y(idx);
	g = g(idx);
	
	% init prediction results
	tp = 0; fp = 0; tn = 0; fn = 0;
	
	% Perform Leave Class Out CV 
	C = max(y); % y must be {1,2,3,...,C}
	for i = 1:C
		if(print_debug)
			fprintf(1,'Processing class %d/%d.\n',i,C);
			if isoctave()
				fflush(stdout);
			end
		end
		% build indices
		testIdx = find(y==i);
		trainIdx = findclasses(y, [1:(i-1), (i+1):C]); 		% see findclasses.m (there's probably a better way around)
		% learn the model (this time: by group!)
		model = fun_train(X(:, trainIdx), g(trainIdx));
		% test the model
		for idx=testIdx
			prediction = fun_predict(model, X(:,idx));
			if(prediction == g(idx))
				tp = tp + 1;
			else
				fp = fp + 1;
			end
		end
	end
	% set result
	validation_result = [tp fp tn fn];
end
