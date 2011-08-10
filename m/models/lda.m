function model = lda(X, y, num_components)
	%% Performs a Linear Discriminant Analysis and returns the 
	%% num_components components sorted descending by their 
	%% eigenvalue. 
	%%
	%% num_components is bound to the number of classes, hence
	%% num_components = min(c-1, num_components)
	%%
	%% Args:
	%%  X: Array with observations given in column.
	%%  y: Classes corresponding to y.
	%%  num_components: Number of components to store.
	%%	
	%% Returns:
	%%  model: Represents the learned model.
	%%
	%% Model description:
	%%  mu - mean of the model.
	%%  name - "lda"
	%%  W - 1:num_components eigenvectors
	%%
	%% Example:
	%% 	lda(I, 200)
	%%
	dim = size(X,1);
	c = max(y); 
	
	if(nargin==2)
		num_components = c - 1
	endif
	
	num_components = min(c-1,num_components);
	
	meanTotal = mean(X,2);
	
	Sw = zeros(dim, dim);
	Sb = zeros(dim, dim);
	for i=1:c
		Xi = X(:,find(y==i));
		meanClass = mean(Xi,2);
		% center data
		Xi = Xi - repmat(meanClass, 1, size(Xi,2));
		% calculate within-class scatter
		Sw = Sw + Xi*Xi';
		% calculate between-class scatter
		Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
	endfor

	% solve the eigenvalue problem
	[V, D] = eig(Sb,Sw);
	
	% sort eigenvectors descending by eigenvalue
	[D,idx] = sort(diag(D),1,'descend');
	V = V(:,idx);
	
	% build model
	model.name = "lda";
	model.num_components = num_components;
	model.W = V(:,1:(c-1));
endfunction
