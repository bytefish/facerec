function model = fisherfaces(X, y, num_components)
	N = size(X,2);
	c = max(y);
	
	% set the num_components
	if(nargin==2)
		num_components=c-1;
	endif
	num_components = min(c-1,num_components);
	
	% reduce dim(X) to (N-c) (see paper [BHK1997])
	Pca = pca(X,y,(N-c));
	Lda = lda(project(X, Pca.W, Pca.mu), y, num_components);
	
	% build model
	model.name = "lda";
	model.mu = repmat(0, size(X,1), 1);
	model.W = Pca.W*Lda.W;
	model.P = model.W'*X;
	model.num_components = Lda.num_components;
	model.y = y;
endfunction
