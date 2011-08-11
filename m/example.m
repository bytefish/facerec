% load function files from subfolders aswell
addpath (genpath ("."));

% load data
[X y width height names] = read_images("/home/philipp/facerec/data/yalefaces_recognition");

%% There's no OOP here. If you want to pass a parameter to the validation, 
%% bind them to the function, see the examples.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Eigenfaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Validation

% Learn Eigenfaces with 100 components
fun_eigenface = @(X,y) eigenfaces(X,y,100);
fun_predict = @(model, Xtest) eigenfaces_predict(model, Xtest, 1);

% a Leave-One-Out Cross Validation (debug)
cv0 = LeaveOneOutCV(X,y,fun_eigenface, fun_predict, 1)
% a 10-fold cross validation (result over all folds, debug)
cv1 = KFoldCV(X,y,10,fun_eigenface, fun_predict, 0, 1)
% a 3-fold cross validation (result over all folds, debug)
cv2 = KFoldCV(X,y,3,fun_eigenface, fun_predict,0,1)

%% Models

% compute a model
eigenface = eigenfaces(X,y,100);

%% Plots

% plot the first (atmost) 16 eigenfaces
figure; hold on;
for i=1:min(16, size(eigenface.W,2))
    subplot(4,4,i);
    comp = cvtGray(eigenface.W(:,i), width, height);
    imshow(comp);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fisherfaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Validation

% Fisherfaces example like Python version
fun_fisherface = @(X,y) fisherfaces(X,y); % no parameters needed
fun_predict = @(model, Xtest) fisherfaces_predict(model, Xtest, 1); % 1-NN

% a Leave-One-Out Cross Validation (debug)
cv0 = LeaveOneOutCV(X,y,fun_fisherface, fun_predict, 1)
% a 10-fold cross validation
cv1 = KFoldCV(X,y,10,fun_fisherface, fun_predict,0,1)
% a 3-fold cross validation
cv2 = KFoldCV(X,y,3,fun_fisherface,fun_predict,0,1)

%% Models

% compute a model
fisherface = fisherfaces(X,y);

% plot fisherfaces
figure; hold on;
for i=1:min(16, size(fisherface.W,2))
    subplot(4,4,i);
    comp = cvtGray(fisherface.W(:,i), width, height);
    imshow(comp);
    title(sprintf("Fisherface #%i", i));
endfor

%% 2D plot of projection (first three classes)
figure; hold on;
for i = findclasses(fisherface.y, [1,2,3])
	text(fisherface.P(1,i), fisherface.P(2,i), num2str(fisherface.y(i)));
endfor

%% 3D plot of projection (first three classes)
figure; hold on;
for i = findclasses(fisherface.y, [1,2,3])
	plot3(fisherface.P(1,i), fisherface.P(2,i), fisherface.P(3,i), 'r.');
	text(fisherface.P(1,i), fisherface.P(2,i), fisherface.P(3,i), num2str(fisherface.y(i)));
endfor

% is a contour plot probably useful?
figure; 
x_values = [1:1:width];
y_values = [1:1:height];
contourf(x_values, y_values, cvtGray(fisherface.W(:,13),width,height));
colorbar;
axis("equal");

