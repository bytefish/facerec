function X = read_image(filename)
	X = [];
	try
		X = double(imread(filename));
		[height width channels] = size(X);
		% greyscale the image if we have 3 channels
		if(channels == 3)
			X = (X(:,:,1) + X(:,:,2) + X(:,:,3)) / 3;
		endif
		X = reshape(X,width*height,1);
	catch
		lerr = lasterror;
		printf("Cannot read image \"%s\".\nReason:\n%s\n", filename, lerr);
	end
endfunction
