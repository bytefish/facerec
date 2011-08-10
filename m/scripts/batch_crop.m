%% configuration

% destination image size
dsize = [70, 70]

%% Celebrities dataset
% size 70,70
% topOffset 0.2
% verticalOffset 0.25

%% Yalefaces A
% size 100, 130
% top 0.4
% left 0.3

% offsets
top = 0.2 % 20% above the eyes
left = 0.25 % 25% of the image as offset to the left and right

% files.txt:
% ./data/c/tom_cruise/01.jpg,236,218,334,219
% ./data/c/tom_cruise/02.jpg,103,144,170,147
% ...

fid = fopen("./files.txt", "r")
while feof(fid) == 0
	l = fgetl(fid);
	r = strsplit(l, ",");
	fn = r{1};
	eye0 = [ str2num(r{2}) str2num(r{3}) ];
	eye1 = [ str2num(r{4}) str2num(r{5}) ];
	crop(r{1}, eye0 , eye1 , top, left, dsize);
endwhile

