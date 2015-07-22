function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feedforward the neural network and return the cost in the variable J.
y1=zeros(m,num_labels);
for i=1:m;
    n=y(i);
    y1(i,n)=1;
end
a1 = [ones(m, 1) X];
a2=sigmoid(a1*Theta1');
a2=[ones(m, 1) a2];
a3=sigmoid(a2*Theta2');

J=1/m*sum(sum(-y1.*log(a3)-(1-y1).*log(1-a3)))+...
    lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% Implement the backpropagation algorithm to compute the gradient Theta1_grad and Theta2_grad. 
delta3=a3-y1;
delta2=delta3*Theta2(:,2:end).*sigmoidGradient(a1*Theta1');
D1=delta2'*a1;
D2=delta3'*a2;

Theta1_grad(:,1) = 1/m*D1(:,1);
Theta1_grad(:,2:end)=1/m*D1(:,2:end)+lambda/m*Theta1(:,2:end);


Theta2_grad(:,1) = 1/m*D2(:,1);
Theta2_grad(:,2:end)=1/m*D2(:,2:end)+lambda/m*Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
