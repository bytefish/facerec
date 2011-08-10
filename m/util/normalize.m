function N = normalize(I, l, h)
	minI = min(I);
	maxI = max(I);
	%% Normalize to [0...1].
	N = I - minI;
	N = N ./ (maxI - minI);
	%% Scale to [low...high].
	N = N .* (h-l);
	N = N + l;
end
