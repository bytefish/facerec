function c = knn(P, y, Q,  k)
	%% k-nearest neighbor classification.
	%%
	%% Args:
	%%  P: Reference vectors given in column.
	%%  Q: Query column vector.
	%%  y: Classes corresponding to P. (y = {1,2,...,n})
	%%	k: Number of nearest neighbors for prediction.
	%%
	%% Returns:
	%%  c: Class identified by the majority of k neighbors.
	%%
	%% Example:
	%%   P=[1,21,20,2,4,30;
	%%      1,21,20,2,4,30]
  %%   y=[1, 3, 3,2,2, 3]
  %%   Q=[1;1]
  %% 
  %%   knn(P,Q,y,1) % returns 1
  %%   knn(P,Q,y,3) % returns 2
	%%   knn(P,Q,y,6) % returns 3
	%%
	n = size(P,2);	
	% clip k
	if (nargin == 3)
	 k=1;
	elseif (k>n)
	 k=n;
	endif

	Q = repmat(Q, 1, n);
	distances = sqrt(sum(power((P-Q),2),1));
	[distances, idx] = sort(distances);
	y = y(idx);
	y = y(1:k);
	h = histc(y,(1:max(y)));
	[v,c] = max(h);
endfunction

%{
P=[1,21,20,2,4,30;
   1,21,20,2,4,30]
y=[1, 3, 3,2,2, 3]
Q=[1;1]

knn(P,Q,y,1) % c == 1
knn(P,Q,y,3) % c == 2
knn(P,Q,y,6) % c == 3
%} 


