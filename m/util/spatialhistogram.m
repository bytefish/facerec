% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function SH = spatialhistogram(L, m, n, numBins)
	%% Calculate the spatial histogram of a LBP image divided into [m x n] cells.
	%% 
	%% Example:
	%%	I=imread("/path/to/image.jpg");
	%%	G = rgb2gray(I);
	%%	L = lbp(G,1,8);
	%%	SH = spatialhistogram(L,8,8,256);
	%%
	%% TODO Uniform Patterns
	%% TODO Vectorize
	%%
	[h,w] = size(L);
	py = uint32(floor(h/m));
	px = uint32(floor(w/n));
	% allocate some memory
	SH = zeros(1,m*n*numBins);
	i = 0;
	for rowIdx=1:m
		for colIdx=1:n
			C = L((rowIdx-1)*py+1:rowIdx*py,(colIdx-1)*px+1:colIdx*px);
			SH(1,(i*numBins+1):((i+1)*numBins)) = hist(C(:), numBins, 1); % normalized histogram
			i+=1;
		end
	end
end
