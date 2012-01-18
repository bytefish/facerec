% load function files from subfolders aswell
addpath (genpath ("."));

% load data
[X y width height names] = read_images("/home/philipp/facerec/data/yalefaces_recognition");

% compute a model
eigenface = eigenfaces(X,y,100);

%% Plots

% plot the first (atmost) 16 eigenfaces
figure; 
title("Eigenfaces (AT&T Facedatabase)");
hold on;
for i=1:min(16, size(eigenface.W,2))
    subplot(4,4,i);
    comp = cvtGray(eigenface.W(:,i), width, height);
    imshow(comp);
    colormap(jet(256));
    title(sprintf("Eigenface #%i", i));
endfor


%% 2D plot of projection (add the classes you want)
figure; hold on;
for i = findclasses(eigenface.y, [1,2,3])
	text(eigenface.P(1,i), eigenface.P(2,i), num2str(eigenface.y(i)));
endfor

%% 3D plot of projection (first three classes, add those you want)
figure; hold on;
for i = findclasses(eigenface.y, [1,2,3])
	plot3(eigenface.P(1,i), eigenface.P(2,i), eigenface.P(3,i), 'r.'); 
	text(eigenface.P(1,i), eigenface.P(2,i), eigenface.P(3,i), num2str(eigenface.y(i)));
endfor

pause;
