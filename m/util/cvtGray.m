% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function G = cvtGray(I, width, height)
	%% Returns a greyscaled representation G of I.
	%%
	%% Args:
	%%  I: Array with width*height elements.
	%%  width: Width of G.
	%%  height: Height of G
	%%
	%% Returns:
	%%  Greyscaled (intensity 0-255) and reshaped image I.
	%%
	%% Example:
	%% 	cvtGray(I, 200, 100)
	%%
	G = reshape(normalize(I, 0, 255), height, width);
	G = uint8(G);
end
