% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function gallery(path, filename, _rows, _cols)
	%% Create a gallery from a given image directory. 
	%%
	%% Args:
	%%  path: Folder to create images from
	%%  filename: filename for the created gallery
	%%  _rows,_cols: rows and columns of the gallery
	%%
	%% Example:
	%%   gallery("/home/philipp/facerec/data/yalefaces_recognition/s03","yale_s01.png",2,6);
	%%
	images = list_files(path);
	[_height, _width,_channels] = size(imread([path,filesep,images{1}]));
	if(nargin < 4)
		_rows = _cols = ceil(sqrt(length(images)));
		if(length(images) <= ((_rows-1)*_cols))
			_rows = _rows-1;
		end
	end
	g = zeros(_rows * _height,_cols * _width, _channels);
	c=1;
	for i=1:_rows
		_row = zeros(_height, _cols*_width,_channels);
  	for j=1:_cols
  		if(c <= length(images))
  			% construct the row as long as we got images to read
				_row(:, ((j-1)*_width+1):(j*_width),:) = imread([path,filesep,images{c}]);
				c=c+1;
			end
		end
		% assign row
		g(((i-1)*_height+1):(i*_height),:,:) = _row;
	end
	imwrite(uint8(g) , filename);
end
