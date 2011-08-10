function idx = findclasses(y, list_of_classes)
	min_class = min(y);
	max_class = max(y);
  idx = [];
  for i = list_of_classes
  	if((i >= min_class) || (i <= max_class))
  		idx = [idx, find(y == i)];
  	endif
  endfor
endfunction
