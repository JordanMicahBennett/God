%% Author ~ Christopher Lu
%% Adaptor ~ Jordan Micah Bennett
%% Adaptor ~ Adaptation Designation ~ Thought Curvature Abstraction : "Causal Neural Perturbation Curvature ( Causal Neural Manifold ( Causal Neural Atom ) )"
%% Adaptor ~ Adaptation Intent : The encodement of curvature of MESOSCALE/MACROSCALE abstraction COMPOSITION, in the Belmanian regime. Therein, I shall derive strictly non-intemperate particle-particle interaction sequences.


%% Defined Constants
number_of_iterations = 500 ; %keep less than 5000
number_of_neurons = 5 ; %keep less than 50, or NaN error
output_learning_rate = .02 ; %should be less than .03 or really 'jumpy'
input_learning_rate = .5 ; %should be less than .5
acceptable_error = .0001; %convergance error

%% Data Input and Standardization
bias = ones(size(aquila_training,1),1);
training_input = [AQUILA_AIRFOIL(1) bias];
average_input = mean(training_input);
stdev_input = std(training_input);
training_input = (training_input(:,:)-average_input(:,1))/stdev_input(:,1

%% Data Target and Standardization
training_target = AQUILA_AIRFOIL(2);
average_output = mean(training_target);
stdev_output = std(training_target);
training_target = (training_target(:,:)-average_output(:,1))/stdev_output(:,1);
training_target = training_target';
%% Allocating Weights and Error Array
inputs = size(training_input,2); % num of weights is twice num of neurons
input_weight = randn(inputs,number_of_neurons)/10; % small weights
output_weight = randn(1,number_of_neurons)/10;
error_plot = zeros(1, number_of_iterations);
%% MAIN LOOP: ANN Algorithim
for i = 1:number_of_iterations
 % secondary loop to evaluate each input/output set
 for j = 1:size(training_input,1)

 n = ceil(rand*size(training_input,1));

 % sigmoid activation function with derivitive = (1-tanh^2)
 activation_function = (tanh(training_input(n,:)*input_weight))';

 % Backpropagation:
 prediction = activation_function'*output_weight';
 error = prediction-training_target(n,1);
 delta_output = error.*output_learning_rate.*activation_function;
 output_weight = output_weight-delta_output';
 delta_input= input_learning_rate.*error.*output_weight'.*(1-
(activation_function.^2))*training_input(n,:); % d/dx tanh
 input_weight = input_weight - delta_input';
 end
 % Visual Output
 prediction = output_weight*tanh(training_input*input_weight)';
 final_error = prediction'-training_target;
 error_plot(i) = (sum(final_error.^2))^0.5;
 figure(1); plot(error_plot)

 % Converged Solution
 if error_plot(i) < acceptable_error
 fprintf('converged after %d iterations.\n',i);
 IN_WT = input_weight;
 OUT_WT = output_weight;
 break
 end
end