%% Author, Meso-scale Simulation Paper: Christopher Lu
%% Author, Thought Curvature Paper: Jordan Micah Bennett 
%% Author, Thought Curvature Paper, Notes: This code structure is designed to take MESOSCALE p^data as parameters to the neural-network model. Training then occurs on this p^data to generate pdata, the empirical distribution w.r.t to the training set of MESOSCALE input space.




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
 %defaultPrediction = activation_function'*output_weight'; %Jordan_note: I recompute the prediction to be the Hamiltonian latent space instead
	
	%Modification by Jordan - For Hamiltonian based backpropagation.
	%I design this psuedo-code based on the account of the Boltzmann Machine, as underlined in 
	%the "Quantum Boltzmann Machine" paper by Amin.
		%construct the quadratic energy distribution
		qed = 0;
		qedNonSquaredComponent_Sum = 0;
		qedSquaredComponent_Sum = 0;
		qedNonSquaredComponent_Sum = qedNonSquaredComponent_Sum + ( bias * training_input(n,:) );
		qedSquaredComponent_Sum = qedSquaredComponent_Sum + ( output_weight * training_input(n,:) * training_input (n,:) ); %(the quadratic part)
		qed = - ( qedNonSquaredComponent_Sum ) - qedSquaredComponent_Sum;
		
		%construct Z
		Z = 0;
		Z = Z + power(qed,-qed);
		
 %Modification by Jordan - this new Hamiltonian has replaced the 'defaultPrediction' above.
 hamiltonianPrediction = power(Z,-1) * Z; 
 
 %Modification by Jordan - To Do's 
 % (1) Develop pseudo-code to complete backpropagation, by updating theta in terms of Hamiltonian based quadratic energy distribution.
 % (2) Develop pseudo-code for the transverse field.
 % (3) Develop pseudo-code for the RL, using learnt numerical simulation data to form a pre-trained Hamiltonian model.
 % (4) Develop pseudo-code for the (Super-) Hamiltonian according to https://arxiv.org/abs/hep-th/0506170
 
 % Toy Example To Do's 
 % CORE-TARGET-SOURCE: https://arxiv.org/pdf/quant-ph/0309022.pdf
 % (1) Perhaps using python, generalize the “Superymmetric LSA” found in the arxiv link above, such that the special unitary matrix Q can capture information independent of sentence reduction.
 % (2) Do a "toy example" on a mnist based dataset, based on the generalization in (1), perhaps using python language
 % (3) Alternatively, find a way to parameterize a special unitary matrix, and compare results to either the unitaryRNN by Martin et al, or the full capacity unitaryRNN by Thomas et al.
 error = hamiltonianPrediction-training_target(n,1);
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
