%% Author, Meso-scale Simulation Paper: Christopher Lu
%% Author, Thought Curvature Paper: Jordan Micah Bennett 
%% Author, Thought Curvature Paper, Notes: The typical neural network model




%% Starting on this level, going down
[Training_Data ,Target_Data, startvalue] = MULTI-RESOLUTION_ANALYSIS(Data);
starting_level = size(outputs,2);
Levels_To_Learn = size(outputs,2);
%% MAIN LOOP
for level = 0:Levels_To_Learn-1
 for i = 1:2^level

 if level == 0 % Initializes Weights
 number_of_inputs = size(inputs,2)+1; % plus bias
 input_weight = randn(number_of_inputs,number_of_neurons)/10; % small weights
 output_weight = randn(1,number_of_neurons)/10;
 else % Extracts and Segments Data from MRA
 seg = (max(Training_Data (:,1))-min(Training_Data (:,1)))/(2^level);
 clear inputs outputs
 cnt = 0;
 section(1) = (seg*(i-1)+min(Training_Data (:,1)));
 section(2) = (seg*(i )+min(Training_Data (:,1)));
 for j = 1:length(Training_Data)
 if (seg*(i-1)+min(Training_Data (:,1))) < Training_Data (j) &&
 Training_Data (j) < (seg*(i)+min(Training_Data (:,1)))
 cnt = cnt+1;
inputs(cnt,1) = Training_Data (j,1);
inputs(cnt,2) = Training_Data (j,2);
outputs(cnt,1) = Target_Data (j,starting_level-level);
 end
 end
 clear input_weight output_weight
 for j = 1:number_of_neurons
 input_weight(:,j) = new_inwt(:,number_of_neurons*(i-1)+j);
 output_weight(1,j) = new_outwt(1,number_of_neurons*(i-1)+j);
 end
 end

 [final, IN_WT, OUT_WT] = ARTIFICIAL_NEURAL_NETWORK(inputs, outputs, &
 input_weight, output_weight, iterations);

 if level == 0 % Allocates Arrays
 ALL_inwts = zeros(2,number_of_neurons);
 ALL_outwts = zeros(1,number_of_neurons);
 new_inwt = zeros(2,number_of_neurons);
 new_outwt = zeros(1,number_of_neurons);
 end
 for j = 1:number_of_neurons % Appends Weights
 ALL_inwts(:,number_of_neurons*(i-1)+j) = IN_WT(:,j);
 ALL_outwts(1,number_of_neurons*(i-1)+j) = OUT_WT(1,j);
 End
 for i = 1:size(ALL_inwts,2) % Expands Weights
 for j=1:number_of_inputs
 new_inwt(j, 2*i-1) = ALL_inwts(j, i);
 new_inwt(j, 2*i) = randn()/10;
 end
 new_outwt(1,2*i-1) = ALL_outwts(1, i);
 new_outwt(1,2*i) = randn()/10;
 end
 end
end
