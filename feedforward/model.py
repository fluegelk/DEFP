import numpy as np
import torch


class TopologyParser:
    """Helper class to parse strings describing the network topology."""

    @staticmethod
    def parse_topology_string(topology_string):
        """
        Parse a string describing the network topology into layers and their parameters.

        Args:
            topology_string: The topology string to parse. Layers are separated by '_', supported layer types are
            convolutional and fully-connveted. Format to describe each layer:
            - for convolutional layers: CONV_{output channels}_{kernel size}_{stride}_{padding}
            - for fully-connected layers: FC_{output units}.

        Returns:
            A list of dicts describing the layer types and parameters
        """
        # Split topology string in tokens and group by layer
        topology_layers = []
        for token in topology_string.split('_'):
            if not any(i.isdigit() for i in token):
                topology_layers.append([])
            topology_layers[-1].append(token)

        # Parse token tuples for each layer and convert to parameter dict
        topology_layers = [TopologyParser.process_layer_description(layer) for layer in topology_layers]
        return topology_layers

    @staticmethod
    def process_layer_description(layer):
        """
        Parse tuple of tokens describing the layer to dict with parameter names.

        Args:
            layer: Tuple of tokens describing this layer, 5-tuple for conv layers, 2-tuple for fc layers.

        Returns:
            Dict containing the layer type and parameters.
        """
        layer_type = layer[0]
        if layer_type == "CONV":
            assert len(layer) == 5
            parameter_names = ["output_channels", "kernel_size", "stride", "padding"]
            return {'type': layer_type,
                    **{key: int(value) for key, value in zip(parameter_names, layer[1:5])}}
        elif layer_type == "FC":
            assert len(layer) == 2
            return {'type': layer_type, 'output_size': int(layer[1])}
        else:
            raise ValueError(f"Layer type {layer_type} not supported")

    @staticmethod
    def weight_initializer(init_name):
        """
        Get the torch initialization function 'torch.nn.init.<init_name>_'. Returns None if the name corresponds to no
        valid initialization function.

        Args:
            init_name: Name of a torch initialization function (without the trailing underscore), e.g. 'normal' for
            torch.nn.init.normal_.

        Returns:
            The torch initialization function, e.g. torch.nn.init.normal_ for init_name 'normal'.
        """
        try:
            return getattr(torch.nn.init, f'{init_name}_')
        except AttributeError:
            return None

    @staticmethod
    def activation(activation_string):
        """
        Get the activation from the given activation name. Invalid names raise a ValueError.

        Args:
            activation_string: The name of the activation, valid names are: 'tanh', 'sigmoid', 'relu', 'none'.

        Returns:
            The torch activation function, e.g. torch.nn.Tanh() for 'tanh'.

        Raises:
            ValueError: on invalid activation names, i.e. not in 'tanh', 'sigmoid', 'relu', 'none'.
        """
        if activation_string == "tanh":
            return torch.nn.Tanh()
        elif activation_string == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation_string == "relu":
            return torch.nn.ReLU()
        elif activation_string == "none":
            return lambda x: x
        else:
            raise ValueError(f'Activation not supported: {activation_string}')


class Model(torch.nn.Module):
    """A neural network consisting of fully-connected and convolutional blocks."""

    def __init__(self, topology, input_size, dropout, convolution_activation, hidden_activation, output_activation,
                 **kwargs):
        """
        Create a new model based on the given topology.
        Args:
            topology: Topology string describing the network topology. To be parsed by the TopologyParser.
            input_size: The input size to the model.
            dropout: Whether to use dropout in the fully connected blocks.
            convolution_activation: The activation to use for the convolutional blocks.
            hidden_activation: The activation to use for the hidden fully-connected blocks.
            output_activation: The activation to use for the output block.
            **kwargs: Additional arguments, unused.
        """
        super().__init__()
        self.dropout = dropout
        self.convolution_activation = convolution_activation
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        topology_layers = TopologyParser.parse_topology_string(topology)
        self.layers = torch.nn.ModuleList()
        self.number_of_layers = len(topology_layers)
        self.number_of_classes = topology_layers[-1]['output_size']

        # Build the network layers
        self._build_model(topology_layers, input_size)

    def _is_output_layer(self, layer_id):
        """
        Returns whether the layer of the given id is the last layer/output layer.

        Args:
            layer_id: The id of the layer to check.

        Returns:
            True iff this is the output layer.
        """
        return layer_id == (self.number_of_layers - 1)

    def _build_model(self, topology_layers, input_size):
        """
        Build the model from the parsed topology layers for the given input size.

        Args:
            topology_layers: A list of descriptions for each layer. The result of TopologyParser.parse_topology_string.
            input_size: The size of the input to the model, e.g. height x width x channels for images. Type: torch.Size
            or similar type.

        Raises:
            ValueError on invalid layer types.
        """
        previous_layer_type = None
        for layer_id, layer in enumerate(topology_layers):
            if layer['type'] == "CONV":
                block, input_size = self._build_convolution_block(layer, input_size)
            elif layer['type'] == "FC":
                # remember first fc layer to know when to transform the input from image to flat array
                if previous_layer_type != "FC":
                    self.first_fc_layer = layer_id
                block, input_size = self._build_fully_connected_block(layer_id, layer, input_size)
            else:
                raise ValueError(f'Invalid layer type {layer["type"]}')
            self.layers.append(block)
            previous_layer_type = layer['type']

    def _build_convolution_block(self, layer, input_size):
        """
        Build convolutional block from its description and input size.

        Args:
            layer: The layer description as returned by the TopologyParser.
            input_size: The input size to this layer (i.e. output size of the previous layer)

        Returns:
            The convolutional block and its output size.
        """
        block = ConvolutionalBlock(input_size[0], layer['output_channels'], layer['kernel_size'], layer['stride'],
                                   layer['padding'], self.convolution_activation)
        output_size = block.compute_output_size(input_size)
        return block, output_size

    def _build_fully_connected_block(self, layer_id, layer, input_size):
        """
        Build fully connected block from its description and input size.

        Args:
            layer_id: The id fo the layer, used to determine if this is the last layer.
            layer: The layer description as returned by the TopologyParser.
            input_size: The input size to this layer (i.e. output size of the previous layer).

        Returns:
            The convolutional block and its output size.
        """
        activation = self.output_activation if self._is_output_layer(layer_id) else self.hidden_activation
        block = FullyConnectedBlock(np.prod(input_size), layer['output_size'], activation, self.dropout)
        return block, torch.Size([layer['output_size']])

    def forward(self, x, error_information=None):
        """
        Define the forward pass: pass the input through the layers, flatten before the first fully connected
        layer, and return the output.

        Args:
            x: The input.
            error_information: Additional error information for the feed-forward training. Unused for conventional
            training with back-propagation.

        Returns:
            The output after passing the input x through all layers.
        """
        for layer_id, layer in enumerate(self.layers):
            if layer_id == self.first_fc_layer:  # transform the input from image to flat array for first FC layer
                x = x.reshape(x.size(0), -1)
            x = layer(x, error_information)
        return x


class FeedForwardModel(Model):
    """
    A neural network for feed-forward-only training consisting of special fully-connected and convolutional blocks for
    feed-forward-only training.
    """

    def __init__(self, topology, input_size, dropout, convolution_activation, hidden_activation, output_activation,
                 feedback_weight_initialization='kaiming_uniform', implementation='true_feed_forward'):
        """
        Create a new feed-forward model based on the given topology.
        Args:
            topology: Topology string describing the network topology. To be parsed by the TopologyParser.
            input_size: The input size to the model.
            dropout: Whether to use dropout in the fully connected blocks.
            convolution_activation: The activation to use for the convolutional blocks.
            hidden_activation: The activation to use for the hidden fully-connected blocks.
            output_activation: The activation to use for the output block.
            feedback_weight_initialization: The initialization to use for the feedback weights. Should correspond to the
            name of a 'torch.nn.init.<init_name>_' function. Passed to TopologyParser.weight_initializer().
            implementation: The underlying implementation to use, either 'true_feed_forward' (to approximate the
            gradients in true ff-only fashion based on a disconnected autograd-graph) or 'gradient_replacement' (to
            replace the actual gradients with the approximated gradients in backward hooks).
        """
        self.feedback_weight_initialization = feedback_weight_initialization
        assert implementation in ['true_feed_forward', 'gradient_replacement']
        self.implementation = implementation
        super().__init__(topology, input_size, dropout, convolution_activation, hidden_activation, output_activation)

    def _build_convolution_block(self, layer, input_size):
        """
        Build convolutional block from its description and input size.

        Args:
            layer: The layer description as returned by the TopologyParser.
            input_size: The input size to this layer (i.e. output size of the previous layer)

        Returns:
            The convolutional block and its output size.
        """
        block = FeedForwardConvolutionalBlock(input_size[0], layer['output_channels'], layer['kernel_size'],
                                              layer['stride'], layer['padding'], self.convolution_activation,
                                              input_size, self.number_of_classes,
                                              weight_initialization=self.feedback_weight_initialization,
                                              implementation=self.implementation)
        output_size = block.compute_output_size(input_size)
        return block, output_size

    def _build_fully_connected_block(self, layer_id, layer, input_size):
        """
        Build fully connected block from its description and input size.

        Args:
            layer_id: The id fo the layer, used to determine if this is the last layer.
            layer: The layer description as returned by the TopologyParser.
            input_size: The input size to this layer (i.e. output size of the previous layer)

        Returns:
            The convolutional block and its output size.
        """
        if self._is_output_layer(layer_id):
            return super()._build_fully_connected_block(layer_id, layer, input_size)
        else:
            block = FeedForwardFullyConnectedBlock(np.prod(input_size), layer['output_size'], self.hidden_activation,
                                                   self.dropout, self.number_of_classes,
                                                   weight_initialization=self.feedback_weight_initialization,
                                                   implementation=self.implementation)
            return block, torch.Size([layer['output_size']])


class FullyConnectedBlock(torch.nn.Module):
    """A fully connected block, consists of a dropout layer, a linear layer, and an activation."""

    def __init__(self, in_features, out_features, activation, dropout, bias=True):
        """
        Create a fully connected block by initializing all its components.

        Args:
            in_features: The in_features of the linear layer.
            out_features: The out_features of the linear layer.
            activation: The activation as string, either 'tanh', 'sigmoid', 'relu', or 'none'.
            dropout: The dropout probability, set to 0 for no dropout.
            bias: Whether to use a bias in the linear layer.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout) if dropout != 0 else lambda x: x
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.activation = TopologyParser.activation(activation)

    def forward(self, x, error_information=None):
        """
        Define the forward pass: pass the input through dropout, linear, and activation and return the output.

        Args:
            x: The input.
            error_information: Additional error information for the feed-forward training. Unused for conventional
            training with back-propagation.

        Returns:
            The output of this block.
        """
        x = self.dropout(x)
        x = self.linear(x)
        return self.activation(x)


class ConvolutionalBlock(torch.nn.Module):
    """A convolutional block, consists of a Conv2d layer, an activation, and pooling (with kernel size 2, stride 2)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, bias=True):
        """
        Create a convolutional block by initializing all its components.

        Args:
            in_channels: The in_channels of the Conv2d layer.
            out_channels: The out_channels of the Conv2d layer.
            kernel_size: The kernel_size of the Conv2d layer.
            stride: The stride of the Conv2d layer.
            padding: The padding of the Conv2d layer.
            activation: The activation as string, either 'tanh', 'sigmoid', 'relu', or 'none'.
            bias: Whether to use a bias in the Conv2d layer.
        """
        super().__init__()
        self.convolution = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.activation = TopologyParser.activation(activation)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, error_information=None):
        """
        Define the forward pass: pass the input through convolution, activation, and pooling and return the output.

        Args:
            x: The input.
            error_information: Additional error information for the feed-forward training. Unused for conventional
            training with back-propagation.

        Returns:
            The output of this block.
        """
        x = self.convolution(x)
        x = self.activation(x)
        return self.pooling(x)

    def compute_output_size(self, input_size, before_pooling=False):
        """
        Compute the output size of the convolutional block for a given input size.

        Args:
            input_size: The input size as torch.Size or similar tuple of the 3 dimensions: [channels, height, width].
            before_pooling: Whether to return the output size after the convolution but before the pooling or after the
            whole block (including pooling) (default).

        Returns:
            The output size as torch.Size.
        """
        assert len(input_size) == 3
        channels, height, width = input_size[0:3]
        assert channels == self.convolution.in_channels

        def side_length(in_length, index):
            conv_output = (in_length - self.convolution.kernel_size[index] + 2 * self.convolution.padding[index]) // \
                          self.convolution.stride[index] + 1  # after conv layer
            pooling_output = conv_output // 2  # after pooling layer
            return conv_output if before_pooling else pooling_output

        return torch.Size([self.convolution.out_channels, side_length(height, 0), side_length(width, 1)])


class FeedForwardFullyConnectedBlock(torch.nn.Module):
    """
    A special fully connected block for feed-forward training. Instead of using the actual gradient, the gradient is
    approximated using feedback weights in place of the actual weights of downstream layers and approximate error
    information to allow updates independent of the forward pass on downstream layers.
    """

    def __init__(self, in_features, out_features, activation, dropout, model_output_size, bias=True,
                 weight_initialization='kaiming_uniform', implementation='true_feed_forward'):
        """
        Create a fully connected block for feed-forward training by initializing all its components.

        Args:
            in_features: The in_features of the linear layer.
            out_features: The out_features of the linear layer.
            activation: The activation as string, either 'tanh', 'sigmoid', 'relu', or 'none'.
            dropout: The dropout probability, set to 0 for no dropout.
            model_output_size: The output size of the model, e.g. the number of classes.
            bias: Whether to use a bias in the linear layer.
            weight_initialization: The initialization to use for the feedback weights. Should correspond to the
            name of a 'torch.nn.init.<init_name>_' function. Passed to TopologyParser.weight_initializer().
            implementation: The underlying implementation to use, either 'true_feed_forward' or 'gradient_replacement'.
        """
        super().__init__()
        assert implementation in ['true_feed_forward', 'gradient_replacement']
        self.implementation = implementation
        self.fc_block = FullyConnectedBlock(in_features, out_features, activation, dropout, bias)
        self.feed_forward_training = FeedForwardTraining([out_features], model_output_size, weight_initialization)

    def forward_with_training(self, x, error_information):
        """
        Define the forward pass with feed-forward-only training: pass the input through the underlying FC block,
        immediately compute the approximate gradient based on the error information, using the feedback weights inplace
        of the actual weights of the downstream layers. Finally, detach and return the output of the underlying block.
        This disconnects the autograd-graph and ensures that any backward calls on downstream layers stop before
        entering this block.

        Idea: Each feed forward block has a separate compute graph by detaching the output tensor. Each block computes
        the gradients within it immediately after the forward step. Instead of the next layer's gradient, we use the
        estimated gradient (based on the feed-forward weights and the error information) as input gradient for backward.
        We can use the optimizer as we would with standard backpropagation. Also, since we usually use a normal (non-ff)
        block for the output layer, we still need to call backward on the loss. The only difference is that
        loss.backward() only computes the gradients for the last block, as each ff-block has already computed its
        gradients in the forward step.

        Args:
            x: The input.
            error_information: Error information for the feed-forward training.

        Returns:
            The detached output after passing the input x through this block.
        """
        x = self.fc_block.forward(x)

        in_gradient = self.feed_forward_training.estimate_gradient(error_information)
        x.backward(in_gradient)

        return x.detach_()  # detach inplace to save memory, also sets requires grad to False

    def forward_with_gradient_replacement(self, x, error_information):
        """
        Define the forward pass with gradient replacement: pass the input through the underlying FC block, followed by
        the FFOnlyTrainingFunction, and return the output.
        In the forward pass, the FFOnlyTrainingFunction has no effect. In the backward pass, it replaces an already
        computed gradient with the approximate gradient based on the feedback weights and error information.
        This method therefore simulates feed-forward-only training by replacing certain gradients (those passed from one
        block to the previous) with the approximate gradient.
        This implementation of FFO training is equivalent to the pytorch implementation provided by Frenkel et al. It
        differs from forward_with_training only in how the gradients are computed. The results are identical.

        Args:
            x: The input.
            error_information: Error information for the feed-forward training.

        Returns:
            The output after passing the input x through this block.
        """
        x = self.fc_block.forward(x)
        x = FFOnlyTrainingFunction.apply(x, error_information, self.feed_forward_training.feedback_weights)
        return x

    def forward_without_training(self, x, error_information):
        """
        Define the forward pass without training during the forward pass (training during the backward pass, e.g. via
        traditional backpropagation is still possible). Pass the input through the underlying FC block and return the
        result.

        Args:
            x: The input.
            error_information: Additional error information, unused.

        Returns:
            The output after passing the input x through this block.
        """
        return self.fc_block.forward(x)

    def forward(self, x, error_information):
        """
        Define the forward pass. Depends on the selected implementation and whether the model is in training mode.

        Args:
            x: The input.
            error_information: Additional error information for the feed-forward training. Unused for inference.

        Returns:
            The output after passing the input x through all layers. The output might be detached (when using
            'true_feed_forward' implementation in train mode).
        """
        if self.training and self.implementation == 'true_feed_forward':
            return self.forward_with_training(x, error_information)
        elif self.training and self.implementation == 'gradient_replacement':
            return self.forward_with_gradient_replacement(x, error_information)
        else:
            return self.forward_without_training(x, error_information)

    def __str__(self):
        """
        String representation of this block as 'FeedForward-<underlying block>'.

        Returns:
            The string representation of this block.
        """
        return f'FeedForward-{str(self.fc_block)}'


class FeedForwardConvolutionalBlock(torch.nn.Module):
    """
    A special convolutional block for feed-forward training. Instead of using the actual gradient, the gradient is
    approximated using feedback weights in place of the actual weights of downstream layers and approximate error
    information to allow updates independent of the forward pass on downstream layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, layer_input_size,
                 model_output_size, bias=True, weight_initialization='kaiming_uniform',
                 implementation='true_feed_forward'):
        """
        Create a convolutional block for feed-forward training by initializing all its components.

        Args:
            in_channels: The in_channels of the Conv2d layer.
            out_channels: The out_channels of the Conv2d layer.
            kernel_size: The kernel_size of the Conv2d layer.
            stride: The stride of the Conv2d layer.
            padding: The padding of the Conv2d layer.
            activation: The activation as string, either 'tanh', 'sigmoid', 'relu', or 'none'.
            layer_input_size: The input size to this block (i.e. output size of the previous block)
            model_output_size: The output size of the model, e.g. the number of classes.
            bias: Whether to use a bias in the Conv2d layer.
            weight_initialization: The initialization to use for the feedback weights. Should correspond to the
            name of a 'torch.nn.init.<init_name>_' function. Passed to TopologyParser.weight_initializer().
            implementation: The underlying implementation to use, either 'true_feed_forward' or 'gradient_replacement'.
        """
        super().__init__()
        assert implementation in ['true_feed_forward', 'gradient_replacement']
        self.implementation = implementation
        self.conv_block = ConvolutionalBlock(in_channels, out_channels, kernel_size, stride, padding, activation, bias)
        output_size = self.compute_output_size(layer_input_size, before_pooling=True)
        self.feed_forward_training = FeedForwardTraining(output_size, model_output_size, weight_initialization)

    def forward_with_training(self, x, error_information):
        """
        Define the forward pass with feed-forward-only training: pass the input through the underlying convolutional
        block, immediately compute the approximate gradient based on the error information, using the feedback weights
        inplace of the actual weights of the downstream layers. Finally, detach and return the output of the underlying
        block. This disconnects the autograd-graph and ensures that any backward calls on downstream layers stop before
        entering this block.

        Idea: Each feed forward block has a separate compute graph by detaching the output tensor. Each block computes
        the gradients within it immediately after the forward step. Instead of the next layer's gradient, we use the
        estimated gradient (based on the feed-forward weights and the error information) as input gradient for backward.
        We can use the optimizer as we would with standard backpropagation. Also, since we usually use a normal (non-ff)
        block for the output layer, we still need to call backward on the loss. The only difference is that
        loss.backward() only computes the gradients for the last block, as each ff-block has already computed its
        gradients in the forward step.

        Args:
            x: The input.
            error_information: Error information for the feed-forward training.

        Returns:
            The detached output after passing the input x through this block.
        """
        x = self.conv_block.convolution(x)
        x = self.conv_block.activation(x)

        in_gradient = self.feed_forward_training.estimate_gradient(error_information)
        x.backward(in_gradient)

        x = x.detach_()  # detach inplace to save memory, also sets requires grad to False
        x = self.conv_block.pooling(x)
        return x

    def forward_with_gradient_replacement(self, x, error_information):
        """
        Define the forward pass with gradient replacement: pass the input through the underlying convolutional block,
        followed by the FFOnlyTrainingFunction, and return the output.
        In the forward pass, the FFOnlyTrainingFunction has no effect. In the backward pass, it replaces an already
        computed gradient with the approximate gradient based on the feedback weights and error information.
        This method therefore simulates feed-forward-only training by replacing certain gradients (those passed from one
        block to the previous) with the approximate gradient.
        This implementation of FFO training is equivalent to the pytorch implementation provided by Frenkel et al. It
        differs from forward_with_training only in how the gradients are computed. The results are identical.

        Args:
            x: The input.
            error_information: Error information for the feed-forward training.

        Returns:
            The output after passing the input x through this block.
        """
        x = self.conv_block.convolution(x)
        x = self.conv_block.activation(x)
        x = FFOnlyTrainingFunction.apply(x, error_information, self.feed_forward_training.feedback_weights)
        x = self.conv_block.pooling(x)
        return x

    def forward_without_training(self, x, error_information):
        """
        Define the forward pass without training during the forward pass (training during the backward pass, e.g. via
        traditional backpropagation is still possible). Pass the input through the underlying convolutional block and
        return the result.

        Args:
            x: The input.
            error_information: Additional error information, unused.

        Returns:
            The output after passing the input x through this block.
        """
        return self.conv_block.forward(x)

    def forward(self, x, error_information):
        """
        Define the forward pass. Depends on the selected implementation and whether the model is in training mode.

        Args:
            x: The input.
            error_information: Additional error information for the feed-forward training. Unused for inference.

        Returns:
            The output after passing the input x through all layers. The output might be detached (when using
            'true_feed_forward' implementation in train mode).
        """
        if self.training and self.implementation == 'true_feed_forward':
            return self.forward_with_training(x, error_information)
        elif self.training and self.implementation == 'gradient_replacement':
            return self.forward_with_gradient_replacement(x, error_information)
        else:
            return self.forward_without_training(x, error_information)

    def compute_output_size(self, input_size, before_pooling=False):
        return self.conv_block.compute_output_size(input_size, before_pooling)


class FeedForwardTraining(torch.nn.Module):
    """
    A module to store the feedback weights and compute approximate gradients based on these feedback weights and
    current error information.
    """

    def __init__(self, layer_output_size, model_output_size, weight_initialization):
        """
        Create a new feed-forward training object and initialize the feedback weights.

        Args:
            layer_output_size: The output size of this block (i.e. the size of the input passed to this module).
            model_output_size: The output size of the model, e.g. the number of classes.
            weight_initialization: The initialization to use for the feedback weights. Should correspond to the
            name of a 'torch.nn.init.<init_name>_' function. Passed to TopologyParser.weight_initializer().
        """
        super().__init__()
        feedback_weight_size = torch.Size([model_output_size, np.prod(layer_output_size)])
        self.feedback_weights = torch.nn.Parameter(torch.Tensor(feedback_weight_size))
        self.layer_output_size = layer_output_size

        self.initialize_feedback_weights(weight_initialization)

    def determine_gradient_shape(self, error_information):
        """
        Determine the correct shape of the gradient (batch-size x layer output size).

        Args:
            error_information: The error information, used to determine the batch-size.

        Returns:
            The correct gradient shape for this block and the current batch.
        """
        return torch.Size([error_information.shape[0], *self.layer_output_size])

    def initialize_feedback_weights(self, initialization):
        """
        Initialize the feedback weights with the specified initialization function and set requires_grad to False.

        Args:
            initialization: The initialization to use for the feedback weights. Should correspond to the  name of a
            'torch.nn.init.<init_name>_' function. Passed to TopologyParser.weight_initializer().
        """
        init_function = TopologyParser.weight_initializer(initialization)
        init_function(self.feedback_weights)
        self.feedback_weights.requires_grad = False

    def estimate_gradient(self, error_information, feedback_weights=None):
        """
        Estimate the approximate gradient based on the given error information and feedback weights.

        Args:
            error_information: Error information for the current batch.
            feedback_weights: Optional feedback weights to use other than self.feedback_weights.

        Returns:
            The approximate gradient.
        """
        gradient_shape = self.determine_gradient_shape(error_information)
        feedback_weights = self.feedback_weights if feedback_weights is None else feedback_weights
        return self.estimate_gradient_static(error_information, feedback_weights, gradient_shape)

    @staticmethod
    def estimate_gradient_static(error_information, feedback_weights, gradient_shape):
        """
        Estimate the approximate gradient based on the given error information and feedback weights. Reshape it to the
        requested gradient shape using view.
        Args:
            error_information: Error information for the current batch, e.g. the gradient of the loss, the error
            (difference between actual and expected output), the sign of the error,...
            feedback_weights: The feedback weights to use instead of the actual downstream weights/gradients.
            gradient_shape: The shape to reshape the computed gradient to (i.e. the shape the actual gradient would
            have). Especially necessary for convolutional layers, as the computed gradient is flat for each sample in
            the batch.

        Returns:
            The approximate gradient.
        """
        # use vector selection when the error information is passed as 1D vector of class indices instead of a
        # matrix multiplication with the one-hot-encoded targets.
        use_vector_selection = (len(error_information.shape) == 1 or error_information.shape[1] == 1)
        if use_vector_selection:
            gradient = feedback_weights[error_information]
        else:  # estimate gradient as matrix multiplication error * weights
            gradient = error_information.mm(feedback_weights)
        return gradient.view(gradient_shape)  # return estimated gradient in correct shape


class FFOnlyTrainingFunction(torch.autograd.Function):
    """
    Function wrapper for the feed forward training to replace the gradient in the backward step. Used for the
    'gradient_replacement' implementation variant.
    """

    @staticmethod
    def forward(ctx, x, error_information, feedback_weights):
        """
        Define the forward pass: save the error information and feedback weights in the context for the backward pass.
        Pass through the input and return it unchanged.

        Args:
            ctx: The context, used to save variables for the backward pass.
            x: The input.
            error_information: Error information for the current batch, used to estimate the approximate gradient.
            feedback_weights: Feedback weights use instead of the actual downstream weights/gradients to estimate the
            approximate gradient.

        Returns:
            The input x, unchanged.
        """
        ctx.save_for_backward(error_information, feedback_weights)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Define the backward pass: load the error information and feedback weights from the context (saved during the
        forward pass). Compute the approximate gradient using the FeedForwardTraining with the error information and
        feedback weights.

        Args:
            ctx: The context, used to load variables from the forward pass.
            grad_output: The gradient passed back from the next layer, used only to get the correct gradient shape.

        Returns:
            3-tuple (gradient, None, None), where gradient is the approximate gradient based on the given error
            information and feedback weights. The remaining two entries in the tuple correspond to the gradients for the
            other two inputs to the forward pass (error_information and feedback_weights) and can be None since they are
            never used.
        """
        error_information, feedback_weights = ctx.saved_tensors
        grad_output = FeedForwardTraining.estimate_gradient_static(error_information, feedback_weights,
                                                                   grad_output.shape)
        return grad_output, None, None


def create_model(algorithm, **kwargs):
    """
    Create a neural network model with the given configuration. Returns either a Model (if 'BP') or FeedForwardModel
    (otherwise) depending on the given training algorithm.

    Args:
        algorithm: The training mode to use, either 'BP' for backpropagation or 'Feed-Forward' for training without
        backpropagation.
        **kwargs: The arguments to the Model or FeedForwardModel initializer.

    Returns:
        The created model, either Model (if BP) or FeedForwardModel (otherwise).
    """
    return Model(**kwargs) if algorithm == 'BP' else FeedForwardModel(**kwargs)
