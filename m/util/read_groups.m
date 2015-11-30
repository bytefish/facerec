% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function [X y g group_names subject_names width height] = read_groups(path)
	%% Read groups of images from a given path and return the Imagematrix X.
	%%
	%% Args:
	%%   path: 
	%%
	%% Returns:
	%%  X [dim x num_data] Array with images given in columns
	%%  y [1 x num_data] Classes corresponding to images of X. 
	%%  g [1 x num_data] Groups corresponding to classes of y.
	%%  group_names {class_idx} names of a group
	%%  subject_names {class_idx} names of the subject of a group
	%%  width: width of the images
	%%  height: height of the images
	%%
	%% Example:
	%% 	[X y width height names] = read_images("./data/yalefaces")
	%%
	%% TODO add fixed image dimension, resizing images if necessary. 
	%%
	folder = list_files(path);
	X = [];
	y = [];
	g = [];
	group_names = {};
	subject_names = {};
	width = 0;
	height = 0;
	n = 0; % to remember class offset (because read_images returns c = {1,2,..,n})
	gi = 1; % need to count if a folder is empty
	for i=1:length(folder)
		subject = folder{i};
		group = [path, filesep, subject];
		% no files to read? -- do nothing
		if(length(list_files(group)) == 0)
			continue;
		end
		% else read group
		[Xi yi width height names] = read_images(group);
		% add class offset
		yi = yi + n;
		% add the data
		X = [X, Xi];
		y = [y, yi];
		g = [g, repmat(gi, 1, size(yi,2))];
		% names
		group_names{gi} = subject;
		subject_names{gi} = names;
		% set new class offset and group
		n = n + max(yi);
		gi = gi+1;
	end
end
