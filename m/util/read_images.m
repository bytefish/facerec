% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function [X y width height names] = read_images(path)
	%% Read images from a given path and return the Imagematrix X.
	%%
	%% Returns:
	%%  X [numDim x numSamples] Array with images given in columns -- [X1,X2,...,Xn]
	%%  y [1 x numSamples] Classes corresponding to images of X. -- [y1,y2,...,yn]
	%%  width [int] width of the images
	%%  height [int] height of the images
	%%  names [cell array] folder name of each class, so names{1} is the name of class 1
	%%
	%% Example:
	%% 	[X y width height names] = read_images("./data/yalefaces")
	%%
	folder = list_files(path);
	X = [];
	y = [];
	names = {};
	n = 1;
	for i=1:length(folder)
		subject = folder{i};
		images = list_files([path, filesep, subject]);
		if(length(images) == 0)
			continue; %% dismiss files or empty folder
		end
   
		added = 0;
		names{n} = subject;
		%% build image matrix and class vector
		for j=1:length(images)
			%% absolute path
			filename = [path, filesep, subject, filesep, images{j}]; 

			%% Octave crashes on reading non image files (uncomment this to be a bit more robust)
			%extension = strsplit(images{j}, "."){end};
			%if(~any(strcmpi(extension, {"bmp", "gif", "jpg", "jpeg", "png", "tiff"})))
			%	continue;
			%endif
      
			% Quite a pythonic way to handle failure.... May blow you up just like the above.
			try
				T = double(imread(filename));
			catch
				lerr = lasterror;
				fprintf(1,'Cannot read image %s', filename)
			end
			
			[height width channels] = size(T);
      
			% greyscale the image if we have 3 channels
			if(channels == 3)
				T = (T(:,:,1) + T(:,:,2) + T(:,:,3)) / 3;
			end
      
			%% finally try to append data
			try
				%% Add image as a column vector:
				X = [X, reshape(T,width*height,1)];
				y = [y, n];
				added = added + 1;
			catch
				lerr = lasterror;
				fprintf(1,'Image cannot be added to the Array. Wrong image size?\n')
			end
		end
		% only increment class if images were actually added!
		if ~(added == 0)
			n = n + 1;
		end
	end
end
