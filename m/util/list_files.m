% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

function L = list_files(path)
	%% List all files in a folder and return it as a cell array.
	%%
	%% Args:
	%%  path: Path to list files from.
	%%
	%% Returns:
	%%  L: Cell array with files in this folder.
	%%
	%% Example:
	%% 	L = list_files("./data/yalefaces")
	L = dir(path);
	L = L(3:length(L));
	L = struct2cell(L);
	L = L(1,:);
end
