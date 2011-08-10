function [X y width height names] = read_images(path)
	%% Read images from a given path and return the Imagematrix X.
	%%
	%% Args:
	%%   path: 
	%%
	%% Returns:
	%%  X: Array with images given in columns -- [X1,X2,...,Xn]
	%%  y: Classes corresponding to images of X. -- [y1,y2,...,yn]
	%%  width: width of the images
	%%  height: height of the images
	%%  names: name of each class -- names{1} is the name of class 1
	%%
	%% Example:
	%% 	[X y width height names] = read_images("./data/yalefaces")
	%%
	%% TODO add fixed image dimension, resizing images if necessary. 
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
    endif
    
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
      
      % Quite a pythonic way to handle a failure.... May blow you up just like the above.
      try
        T = double(imread(filename));
      catch
	      lerr = lasterror;
	      printf("Cannot read image \"%s\"", filename)
			end
			
      [height width channels] = size(T);
      
      % greyscale the image if we have 3 channels
      if(channels == 3)
         T = (T(:,:,1) + T(:,:,2) + T(:,:,3)) / 3;
      endif
      
      %% finally try to append data
      try
        X = [X, reshape(T,width*height,1)];
        y = [y, n];
        added = added + 1;
      catch
        lerr = lasterror;
        printf("Image cannot be added to the Array. Wrong image size?\n")
      end
      fflush(stdout); % show warnings (probably not necessary, doesn't harm anyway)
    endfor
    % only increment class if images were actually added!
    if ~(added == 0)
    	n = n + 1;
    endif
  endfor
endfunction
