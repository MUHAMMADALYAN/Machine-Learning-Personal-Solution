function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(1,size(X, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


                      %My explaination

% every row of Theta is best theta value for crosseponding label 1-> 1 ,2 row -> 2nd label e.g. first row of theta best for label1(cat
% we multiply each rows of Theta with every example and for a particular example in which theta gives best prediction value out of all
% we see index of that theta and that is equal to specific label 
a1=X;
a1 = [ones(m, 1) X];
a2=sigmoid(a1*Theta1'); #Theta1 contain best value for neurons e.g. --> neuron1...neuron28
%now each row (has one  training example * withall thetas) we will pick those theta who gives us max value and its index is 

a2=[ones(m,1) a2];
a3=sigmoid(Theta2*a2');%a2 is 100x28 theta 10x28  
[i,p]=max(a3); % this gives us row matrix of predictions we need column so take transpose
p=p';







% =========================================================================


end
