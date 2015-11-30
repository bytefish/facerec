% Copyright (c) Philipp Wagner. All rights reserved.
% Licensed under the BSD license. See LICENSE file in the project root for full license information.

% load function files from subfolders aswell
addpath (genpath ('.'));

% load data
[X y width height names] = read_images('D:/facerec/data/at');

% Compute the model (full PCA):
eigenface = eigenfaces(X,y);

% Plot the first (atmost) 16 eigenfaces:
figure; hold on;
title('Eigenfaces (AT&T Facedatabase)');
for i=1:min(16, size(eigenface.W,2))
  subplot(4,4,i);
  comp = cvtGray(eigenface.W(:,i), width, height);
  imshow(comp);
  colormap(jet(256)); % Comment this out, if you don't want colored images
  title(sprintf('Eigenface #%i', i));
end

%% 2D plot of projection (add the classes you want):
figure; hold on;
for i = findclasses(eigenface.y, [1,2,3])
  text(eigenface.P(1,i), eigenface.P(2,i), num2str(eigenface.y(i)));
end

%% 3D plot of projection (first three classes, add those you want):
if(size((eigenface.P),2) >= 3)
  figure; hold on;
  for i = findclasses(eigenface.y, [1,2,3])
    % LineSpec: red dots 'r.'
    plot3(eigenface.P(1,i), eigenface.P(2,i), eigenface.P(3,i), 'r.'), view(45,-45);
    text(eigenface.P(1,i), eigenface.P(2,i), eigenface.P(3,i), num2str(eigenface.y(i)));
  end
end

%% Plot eigenfaces reconstruction
steps = 10:20:min(eigenface.num_components,320) ;
Q = X (:,1) ; % first image to reconstruct (each image is a column!)
figure;
title ('Reconstruction (AT&T Facedatabase)');
hold on;
for i =1:min(16,length(steps))
  subplot (4, 4, i);
  numEvs = steps(i);
  P = project(Q, eigenface.W(:,1:numEvs), eigenface.mu);
  R = reconstruct(eigenface.W(:,1:numEvs), P, eigenface.mu);
  comp = cvtGray(R, width, height);
  imshow(comp);
  title(sprintf('%i Eigenvectors', numEvs));
end

pause;
