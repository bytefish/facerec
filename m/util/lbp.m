% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function L = lbp(X, radius, neighbors) 
	%% Description:
	%%	Calculates the Extended Local Binary Patterns of X with given radius 
	%%	and neighbors. This is a slightly modified version of: 
	%%	http://www.cse.oulu.fi/CMV/Downloads/LBPMatlab by authors Marko Heikkilä 
	%%	and Timo Ahonen.
	%%
	%% Arguments:
	%%	radius [int] 
	%%	neighbors [int]
	%%
	%% Example:
	%%	I=imread("/path/to/image.jpg");
	%%	G=rgb2gray(I);
	%%	L=lbp(G);
	%%	imshow(uint8(L))
	%%
	X = double(X);
	% get origin of a block
	origy = radius+1;
	origx = radius+1;
	% blocks to process
	dy = size(X,1)-(2*radius+1);
	dx = size(X,2)-(2*radius+1);
	% create result matrix
	L = zeros(dy+1,dx+1);
	% get center matrix
	C = X(origy:origy+dy,origx:origx+dx);
	% iterate through circle
	for n=1:neighbors
		% sample points
		y = radius * -sin(2*pi*((n-1)/neighbors)) + origy;
		x = radius * cos(2*pi*((n-1)/neighbors)) + origx;
		% relative indices
		fx = floor(x);
		fy = floor(y);
		cx = ceil(x);
		cy = ceil(y);
		% fractional parts
		tx = x - fx;
		ty = y - fy;
		% interpolation weights
		w1 = (1 - tx) * (1 - ty);
		w2 =      tx  * (1 - ty);
		w3 = (1 - tx) *      ty ;
		w4 =      tx  *      ty ;
		% get interpolated image
		N = w1*X(fy:fy+dy,fx:fx+dx) + w2*X(fy:fy+dy,cx:cx+dx) + w3*X(cy:cy+dy,fx:fx+dx) + w4*X(cy:cy+dy,cx:cx+dx);
		% calculate binary value for current neighbor, update result
		L += (N>=C)*(2^(n-1));
	end
end
