-module(mcp).

-export([connect/2, neuron/4, stock_market/0, trainer/5,
	 xor_problem/0]).

neuron(Weights, Inputs, Connected_deltas,
       Trainer_PID) ->
    Sigmoid = fun (X) -> 1 / (1 + math:exp(-X)) end,
    Sigmoid_prime = fun (X) ->
			    math:exp(-X) / (1 + math:exp(-2 * X))
		    end,
    receive
      % wait for a stimulus message from another neuron
      % with the message paramater of new_input
      {stimulus, New_input} ->
	  % change the input coming from this stimulus's neuron
	  New_inputs = change_val(Inputs, New_input),
	  % recalculate the output of this neuron
	  % need to extract inputs because the format of
	  % a single inputs is a tuple of {pid, input}
	  % this is so you can identify who this input came from
	  Output = output(Sigmoid, Weights,
			  extract_inputs(New_inputs)),
	  % this will send your calculated output to all of
	  % the neurons you are connected to
	  send_val(self(), Output, extract_keys(Connected_deltas),
		   Trainer_PID),
	  % Needed so the inputs and weights of this neuron
	  % persist in a stateless enviornment
	  neuron(Weights, New_inputs, Connected_deltas,
		 Trainer_PID);
      % Add new neurons FROM which this neuron branches to
      {add_output, Output_pid} ->
	  % Add the output neurons pid to the connected_neurons list
	  New_connected_deltas = [{Output_pid, 1}
				  | Connected_deltas],
	  %save state
	  neuron(Weights, Inputs, New_connected_deltas,
		 Trainer_PID);
      %Add new input connections TO this neuron
      {add_input, Input_pid} ->
	  % add the pid to the inputs list, and add some arbitrary input value
	  % for the time being, but can be changed with a stimulus call using
	  % the Input_pid neuron
	  New_inputs = [{Input_pid, 1.0} | Inputs],
	  % also you have to add a new weights entry for this neuron.
	  % added some arbitrary weight. but can always be changed with
	  % learning.
	  New_Weights = [1.0 | Weights],
	  %save state
	  neuron(New_Weights, New_inputs, Connected_deltas,
		 Trainer_PID);
      % this message will make this neuron just be a buffer neuron
      % thus just passing this Input to all of its Connected Neurons
      {buffer, Input} ->
	  %Forward input just like any other buffer circuit element
	  send_val(self(), Input, extract_keys(Connected_deltas),
		   Trainer_PID),
	  neuron(Weights, Inputs, Connected_deltas, Trainer_PID);
      % back propogation learning algorithm implementation
      {learn, Backprop_signal} ->
	  Ita = 1,
	  Modified_deltas = update_delta(Connected_deltas,
					 Backprop_signal),
	  Orig_output = output(Sigmoid, Weights,
			       extract_inputs(Inputs)),
	  Prime_output = output(Sigmoid_prime, Weights,
				extract_inputs(Inputs)),
	  Delta = calc_delta(Backprop_signal, Inputs,
			     Modified_deltas, Orig_output, Prime_output),
	  Weight_deltas = [neuron_1(V1, Delta, Ita)
			   || V1 <- extract_inputs(Inputs)],
	  Updated_weights = [Weight + Delta
			     || {Weight, Delta}
				    <- lists:zip(Weights, Weight_deltas)],
	  io:format("Neuron ~w has updated weights from ~w "
		    "to ~w~n",
		    [self(), Weights, Updated_weights]),
	  % Propogate the delta times weight into the delta to make
	  % calculations easier for the new delta calcs
	  [Input_neuron ! {learn, {self(), Delta * Weight}}
	   || {Input_neuron, Weight}
		  <- lists:zip(extract_keys(Inputs), Weights)],
	  % save state
	  neuron(Updated_weights, Inputs, Modified_deltas,
		 Trainer_PID);
      % save state with a real trainer object this time.
      {attach_trainer, Trainer_Attach} ->
	  neuron(Weights, Inputs, Connected_deltas,
		 Trainer_Attach)
    end.

neuron_1(Input, Delta, Ita) -> Ita * Delta * Input.

trainer(Input_pairs, Desired_outputs, Output_PID,
	Input_PIDs, Bias_PID) ->
    receive
      {start_next} ->
	  [Input_set | Sliced_inputs] = Input_pairs,
	  [begin
	     Input_node ! {buffer, New_input_single},
	     Bias_PID ! {buffer, 1}
	   end
	   || {New_input_single, Input_node}
		  <- lists:zip(Input_set, Input_PIDs)],
	  trainer(Sliced_inputs, Desired_outputs, Output_PID,
		  Input_PIDs, Bias_PID);
      {learn} ->
	  [Desired_curr | Sliced_desired] = Desired_outputs,
	  Output_PID ! {learn, {Output_PID, Desired_curr}},
	  trainer(Input_pairs, Sliced_desired, Output_PID,
		  Input_PIDs, Bias_PID)
    end.

% update the deltas or set to [] if empty
update_delta(Deltas, Delta) ->
    case Deltas of
      [] -> [];
      Delta_hash when is_list(Delta_hash) ->
	  change_val(Delta_hash, Delta)
    end.

% input node
calc_delta(Backprop, Inputs, Deltas, Output,
	   Prime_output)
    when Deltas =/= [], Inputs =:= [] ->
    null;
% output node
calc_delta(Backprop, Inputs, Deltas, Output,
	   Prime_output)
    when Deltas =:= [], Inputs =/= [] ->
    % the sensitiviy for the output node is really the desired value given by
    % the trainer to the whole nn
    {_, Real_val} = Backprop,
    (Real_val - Output) * Prime_output;
% hidden node
calc_delta(Backprop, Inputs, Deltas, Output,
	   Prime_output)
    when Deltas =/= [], Inputs =/= [] ->
    lists:sum(extract_inputs(Deltas)) * Prime_output.

% main output of neurons
output(Nonlinear, Weights, Inputs) ->
    Nonlinear(lists:sum([W * I
			 || {W, I} <- lists:zip(Weights, Inputs)])).

% this will search through inputs
% find where pid is equal to inputs[i][1]
% if it finds the tuple that satisfies that
% it will replace that entire tuple with input
% thus, changing the input value associated
% with the pid of that neuron
change_val(Inputs, Input) ->
    {Pid, _} = Input,
    lists:keyreplace(Pid, 1, Inputs, Input).

% this will just extract the last value of the "input" tuple
% into a list as the first value is just the pid of the input neuron
% and the second value is the actual input value
extract_inputs(Inputs) ->
    [extract_inputs_1(V1) || V1 <- Inputs].

extract_inputs_1(Input) ->
    {_, Real_val} = Input, Real_val.

%% extract the keys
extract_keys(Inputs) ->
    [extract_keys_1(V1) || V1 <- Inputs].

extract_keys_1(Input) -> {Key, _} = Input, Key.

send_val(Self_pid, Value, Neurons, Trainer_PID) ->
    case Neurons of
      [] ->
	  io:format("Final output of neuron ~w  is ~w~n",
		    [Self_pid, Value]),
	  Trainer_PID ! {learn};
      Connections when is_list(Connections) ->
	  %io:format("Sent output of ~w to ~w~n", [Self_pid, Neurons]),
	  [send_val_1(V1, Self_pid, Value) || V1 <- Connections]
    end.

send_val_1(Neuron_pid, Self_pid, Value) ->
    Neuron_pid ! {stimulus, {Self_pid, Value}}.

% fully connect two neurons together
connect(Send_neuron, Receive_neuron) ->
    % connect so that Send_neuron is an input for Receive_neuron
    Receive_neuron ! {add_input, Send_neuron},
    % connect so that the Receive_neuron is an output for Send_neuron
    Send_neuron ! {add_output, Receive_neuron}.

spawnlayer(Size) ->
    [spawn(mcp, neuron, [[], [], [], self()])
     || _ <- lists:seq(1, Size)].

connectlayers(Layer1, Layer2) ->
    [connect(L1, L2) || L1 <- Layer1, L2 <- Layer2].

xor_problem() ->
    Bias = spawnlayer(1),
    InputLayer = spawnlayer(2),
    H1 = spawnlayer(2),
    OutputLayer = spawnlayer(1),
    connectlayers(Bias, InputLayer),
    connectlayers(Bias, H1),
    connectlayers(InputLayer, H1),
    connectlayers(H1, OutputLayer),
    Training_Input = [[0, 0], [0, 1], [1, 0], [1, 1]],
    Training_Desired = [0, 1, 1, 0],
    MrBuu = spawn(mcp, trainer,
		  [Training_Input, Training_Desired, hd(OutputLayer),
		   InputLayer, hd(Bias)]),
    hd(OutputLayer) ! {attach_trainer, MrBuu},
    [MrBuu ! {start_next}
     || _ <- lists:seq(1, length(Training_Desired))],
    [X1 , X2] = InputLayer,
    
    X1 ! {buffer, 0},
    X2 ! {buffer, 0},
    io:format("Should be 0"),
    
    X1 ! {buffer, 0},
    X2 ! {buffer, 1},
    io:format("Should be 1"),

    X1 ! {buffer, 1},
    X2 ! {buffer, 0},
    io:format("Should be 1"),

    X1 ! {buffer, 1},
    X2 ! {buffer, 1},
    io:format("Should be 0").


stock_market() ->
    Bias = spawnlayer(1),
    InputLayer = spawnlayer(4),
    L1 = spawnlayer(100),
    L2 = spawnlayer(100),
    L3 = spawnlayer(100),
    OutputLayer = spawnlayer(1),
    connectlayers(Bias, InputLayer),
    connectlayers(Bias, L1),
    connectlayers(Bias, L2),
    connectlayers(Bias, L3),
    connectlayers(InputLayer, L1),
    connectlayers(L1, L2),
    connectlayers(L2, L3),
    connectlayers(L3, OutputLayer),
    Training_Input = [[7.61747e+3, 7.61747e+3,
		       7.43793999999999959982e+3, 5116380000]],
    Training_Desired = [7.47263e+3],
    Inspector = spawn(mcp, trainer,
		      [Training_Input, Training_Desired, hd(OutputLayer),
		       InputLayer, hd(Bias)]),
    hd(OutputLayer) ! {attach_trainer, Inspector},
    [Inspector ! {start_next}
     || _ <- lists:seq(1, length(Training_Desired))].
