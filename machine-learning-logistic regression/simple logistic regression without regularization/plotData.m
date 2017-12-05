function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


x_pos_index = y>0;
X1 = X(x_pos_index,:);
plot(X1(:,1), X1(:,2), 'k+','color','black')
hold on
X2 = X(~x_pos_index,:);
plot(X2(:,1),X2(:,2),'ko','color','yellow')





% =========================================================================



hold off;

end
