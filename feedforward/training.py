import logging
import time

import pandas as pd
import torch
import torch.nn.functional
from tqdm import tqdm

logger = logging.getLogger("DEFP")


class TrainingDataCollection:
    """Collect training data (both constant data and data for each epoch) and convert it to a pandas dataframe."""

    def __init__(self):
        """Create a new, empty data collection object."""
        self.constant_columns = {}
        self.epoch_data = []

    def add_epoch_data(self, epoch_data_row):
        """
        Add epoch data to the collected data.

        Args:
            epoch_data_row: The epoch data as dictionary describing one row of the final data frame (excluding constant
            columns).
        """
        self.epoch_data.append(epoch_data_row)

    def add_constant_columns(self, constant_columns):
        """
        Add one or multiple constant columns to the data, i.e. key-value pairs that are the same for all epochs.

        Args:
            constant_columns: The constant columns as dictionary, each key-value pair describing one column.
        """
        self.constant_columns = {**self.constant_columns, **constant_columns}

    def build_dataframe(self):
        """
        Convert the collected data to a pandas dataframe.

        Returns:
            The resulting pandas dataframe.
        """
        dataframe = pd.DataFrame(self.epoch_data)
        for name, value in self.constant_columns.items():
            dataframe[name] = value
        return dataframe

    def reset(self):
        """Reset the collected data by deleting all epoch data and all constant columns."""
        self.constant_columns = {}
        self.epoch_data = []


def create_loss_function(loss_name):
    """
    Create a loss function from its name by creating a wrapper to the corresponding torch.nn.functional function (e.g.
    torch.nn.functional.mse_loss for 'MSE') and preparing the targets if necessary.

    Args:
        loss_name: The name of the loss function to create, one of: 'MSE', 'BCE', or 'CE'.

    Returns:
        The loss function as (outputs, targets, *args, **kwargs) -> loss value. The *args and **kwargs are optional
        arguments to the corresponding torch.nn.functional function.

    Raises:
        ValueError: on an invalid loss name (i.e. not in 'MSE', 'BCE', 'CE').
    """
    loss_functions = {'MSE': (torch.nn.functional.mse_loss, None),
                      'BCE': (torch.nn.functional.binary_cross_entropy, None),
                      'CE': (torch.nn.functional.cross_entropy, lambda l: torch.max(l, 1)[1])}
    if loss_name not in loss_functions:
        raise ValueError(f'Invalid loss {loss_name}. Valid losses are: {", ".join(loss_functions.keys())}.')
    loss, target_preparation = loss_functions[loss_name]

    def loss_function(outputs, targets, *args, **kwargs):
        if target_preparation is not None:
            targets = target_preparation(targets)
        return loss(outputs, targets, *args, **kwargs)

    return loss_function


def create_optimizer(optimizer_name, model, **kwargs):
    """
    Create a torch optimizer of the specified type for the given model.

    Args:
        optimizer_name: The type of optimizer to create (as string), one of: 'SGD', 'NAG', 'Adam', 'RMSprop'.
        model: The model to train with the optimizer.
        **kwargs: Additional named parameters to the optimizers init function.

    Returns:
        The optimizer.

    Raises:
        ValueError: on an invalid loss name (i.e. not in 'SGD', 'NAG', 'Adam', 'RMSprop').
    """
    optimizers = {'SGD': (torch.optim.SGD, {'momentum': 0.9, 'nesterov': False}),
                  'NAG': (torch.optim.SGD, {'momentum': 0.9, 'nesterov': True}),
                  'Adam': (torch.optim.Adam, {}),
                  'RMSprop': (torch.optim.RMSprop, {})}
    if optimizer_name not in optimizers:
        raise ValueError(f'Invalid optimizer {optimizer_name}. Valid optimizers are: {", ".join(optimizers.keys())}.')

    optimizer, init_kwargs = optimizers[optimizer_name]
    return optimizer(model.parameters(), **init_kwargs, **kwargs)


def create_learning_rate_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Create a torch learning rate scheduler of the specified type for the given optimizer.

    Args:
        scheduler_name: The type of learning rate scheduler to create (as string), must be a valid scheduler name
        from torch.optim.lr_scheduler, e.g. 'ExponentialLR'.
        optimizer: The optimizer to use this scheduler with.
        **kwargs: Additional named parameters to the schedulers init function.

    Returns:
        The learning rate scheduler.

    Raises:
        ValueError: on an invalid scheduler name (i.e. no class in torch.optim.lr_scheduler).
    """
    try:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
    except AttributeError:
        raise ValueError(f'Invalid learning rate scheduler {scheduler_name}. Please pass a valid scheduler name from '
                         'torch.optim.lr_scheduler')
    return scheduler_class(optimizer, **kwargs)


class Training:
    """
    Training class for one training run on a given model, including evaluation and data collection after each epoch.
    """
    def __init__(self, model, loss, optimizer, lr_scheduler, device, train_loader, traintest_loader, test_loader,
                 error_information='targets', number_of_classes=None, verbose=0):
        """
        Create a new training object to train the model on the given data using the specified loss, optimizer, and
        (optional) learning rate scheduler.

        Args:
            model: The model to train.
            loss: The loss to train with.
            optimizer: The loss to use.
            lr_scheduler: The learning rate scheduler to use.
            device: The torch device to train on.
            train_loader: Data loader for the training data.
            traintest_loader: Data loader for the train-test data (i.e. for evaluation on the training data).
            test_loader: Data loader for the test data.
            error_information: The error information to use. One of 'targets', 'error', 'error_sign', 'delayed_error',
            'delayed_loss'.
            number_of_classes: The number of classes (used to prepare the target labels).
            verbose: The level of verbosity. The available levels are: <1 (no progress bars), >=1 (training progress),
            >=2 (epoch progress), and >=3 (both training and epoch progress as nested progress bars).
        """
        self.model = model
        self.model.to(device)
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.train_loader = train_loader
        self.traintest_loader = traintest_loader
        self.test_loader = test_loader

        self.delayed_error_information = None
        self.error_information = error_information
        self.number_of_classes = number_of_classes

        self.results = TrainingDataCollection()
        max_memory_allocated = torch.cuda.max_memory_allocated(self.device)
        max_memory_reserved = torch.cuda.max_memory_reserved(self.device)
        self.results.add_constant_columns({'initial_memory_allocated': max_memory_allocated,
                                           'initial_memory_reserved': max_memory_reserved})

        self.tqdm_training_config, self.tqdm_epoch_config = {}, {}
        self.configure_verbosity(verbose)

    def configure_verbosity(self, verbose):
        """
        Prepare tqdm progress bar configurations according to the selected verbosity.
        Args:
            verbose: The level of verbosity. The available levels are: <1 (no progress bars), >=1 (training progress),
            >=2 (epoch progress), and >=3 (both training and epoch progress as nested progress bars).
        """
        progress_bar_per_trial = verbose == 1 or verbose > 2
        progress_bar_per_epoch = verbose > 1
        nested_progress_bars = progress_bar_per_epoch and progress_bar_per_trial

        self.tqdm_training_config = {'disable': not progress_bar_per_trial}
        self.tqdm_epoch_config = {'disable': not progress_bar_per_epoch, 'leave': not nested_progress_bars}

    def init_error_information(self):
        """Initialize the tensor to store the delayed error information with ones if required."""
        if 'delayed' in self.error_information:
            self.delayed_error_information = torch.ones((len(self.train_loader.dataset), self.number_of_classes),
                                                        device=self.device)

    def get_error_information(self, label, label_one_hot, sample_indices):
        """
        Get the error information depending on the selected type of error information (specified in
        self.error_information).

        Args:
            label: The labels for the current batch.
            label_one_hot: The labels with one-hot encoding.
            sample_indices: The index of the samples in the current batch, used only for delayed errors.

        Returns:
            The error information depending on the selected type of error information:
            - for delayed error and loss: the delayed error information saved of these samples and classes
            - for targets: the labels themself
            - for everything else: the one-hot encoded labels.
        """
        if 'delayed' in self.error_information:
            return label_one_hot * self.delayed_error_information[sample_indices].detach()
        elif self.error_information == 'targets':
            return label
        else:
            return label_one_hot

    def update_error_information(self, output, targets, sample_index):
        """
        Update the delayed error information with the results of the current batch by setting the entries for the
        samples in the current batch to either the error or the gradient of the loss for the current output.

        Args:
            output: The model output for the current batch.
            targets: The expected targets for the current batch.
            sample_index: The index of the samples in the current batch.
        """
        if self.error_information == 'delayed_loss':
            self.delayed_error_information[sample_index] = -self.compute_loss_gradient(output, targets)
        elif self.error_information == 'delayed_error':
            self.delayed_error_information[sample_index] = targets - output

    def train(self, epochs, train_mode):
        """
        Train the model for a given number of epochs using the specified training mode.
        Args:
            epochs: The number of epochs to train.
            train_mode: The training mode to use, either 'BP' for backpropagation or 'Feed-Forward' for training without
            backpropagation.
        """
        self.init_error_information()

        for epoch in tqdm(range(epochs), **self.tqdm_training_config):
            current_epoch = epoch + 1
            logger.info(f"------ Epoch {current_epoch:{len(str(epochs))}} / {epochs} ------")

            start = time.perf_counter()
            self.train_epoch(train_mode)
            end = time.perf_counter()

            epoch_results = self.add_epoch_results(current_epoch, end - start)
            self.lr_scheduler_step(epoch_results['test_loss'])

    def get_current_learning_rate(self):
        """
        Get the current learning rate.

        Returns:
            The current learning rate of the optimizer.
        """
        return self.optimizer.param_groups[0]['lr']

    def lr_scheduler_step(self, test_loss):
        """
        Perform a learning rate scheduler step. If the scheduler is a ReduceLROnPlateau scheduler, the given test loss
        is passed as metric to the step() method of the scheduler.

        Args:
            test_loss: The current test loss, passed to scheduler.step() if the scheduler is a
            torch.optim.lr_scheduler.ReduceLROnPlateau. Ignored otherwise.
        """
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(test_loss)
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def add_epoch_results(self, current_epoch, training_time):
        """
        Evaluate the current state of the model and training. Log the results and add them to collection of results.
        The evaluated metrix include: the peak memory allocated and reserved on the cuda device, the train and test
        loss and accuracy, the running time of the epoch, the current learning rate, and the epoch number.

        Args:
            current_epoch: Which epoch is evaluated.
            training_time: The running time of the evaluated epoch.

        Returns:
            The dictionary containing all epoch results.
        """
        max_memory_allocated = torch.cuda.max_memory_allocated(self.device)
        max_memory_reserved = torch.cuda.max_memory_reserved(self.device)
        logger.info(f"Memory Stats: {max_memory_allocated} (allocated) | {max_memory_reserved} (reserved)")

        # Compute loss and accuracy on training and testing set
        train_loss, train_accuracy = self.evaluate_model(self.traintest_loader, 'Train')
        test_loss, test_accuracy = self.evaluate_model(self.test_loader, 'Test')

        epoch_results = {'epoch': current_epoch,
                         'time': training_time,
                         'lr': self.get_current_learning_rate(),
                         "train_accuracy": train_accuracy,
                         "test_accuracy": test_accuracy,
                         "train_loss": train_loss,
                         "test_loss": test_loss,
                         "max_memory_allocated": max_memory_allocated,
                         "max_memory_reserved": max_memory_reserved}

        self.results.add_epoch_data(epoch_results)
        return epoch_results

    def prepare_targets(self, label):
        """
        Reshape to labels to the expected target shape (i.e. one-hot-encoding for classification tasks)

        Args:
            label: The label to reshape.

        Returns:
            The targets in the expected shape.
        """
        targets = torch.zeros(label.shape[0], self.number_of_classes, device=self.device)
        return targets.scatter_(1, label.unsqueeze(1), 1.0)

    def train_epoch(self, train_mode):
        """
        Train the model for one epoch. Select the specific training function based on the provided training mode (either
        train_epoch_feed_forward_only (for training mode 'Feed-Forward') or train_epoch_with_back_propagation (for
        training mode 'BP').

        Args:
            train_mode: The training mode to use, either 'BP' for backpropagation or 'Feed-Forward' for training without
            backpropagation.
        """
        self.model.train()
        if train_mode == 'Feed-Forward':
            self.train_epoch_feed_forward_only()
        else:
            self.train_epoch_with_back_propagation()

    def train_epoch_with_back_propagation(self):
        """
        Train the model for one epoch with backpropagation, i.e. iterate over all batches, perform a forward pass to
        compute the model output, compute the loss of that output, perform a backward pass to compute the gradients,
        and optimizer the model parameters.
        """
        for _, data, label in tqdm(self.train_loader, **self.tqdm_epoch_config):
            data, label = data.to(self.device), label.to(self.device)
            targets = self.prepare_targets(label)

            self.optimizer.zero_grad()
            output = self.model(data, None)
            loss_val = self.loss(output, targets)
            loss_val.backward()
            self.optimizer.step()

    def train_epoch_feed_forward_only(self):
        """
        Train the model for one epoch with a feed forward only approach. This differs from
        train_epoch_with_back_propagation as follows:
        - the error information is computed based on the input data and passed to the model in the forward function. It
          can than be used to update the model during the forward step.
        - after each batch, the stored error information is updated with the current error/loss (when using a delayed
          error).
        """
        for sample_index, data, label in tqdm(self.train_loader, **self.tqdm_epoch_config):
            data, label = data.to(self.device), label.to(self.device)
            targets = self.prepare_targets(label)
            error_information = self.get_error_information(label, targets, sample_index)

            self.optimizer.zero_grad()
            output = self.model(data, error_information)
            loss_val = self.loss(output, targets)
            loss_val.backward()  # backward is still necessary for the output layer (since it is a normal, non-ff block)
            self.optimizer.step()

            self.update_error_information(output, targets, sample_index)

    def compute_loss_gradient(self, output, targets):
        """
        Compute the gradient of the loss wrt. the output.

        Args:
            output: The model output for the current batch.
            targets: The targets for the current batch (i.e. the expected output).

        Returns:
            The gradient of the loss wrt. the output.
        """
        output = output.detach()
        output.requires_grad = True
        loss_val = self.loss(output, targets, reduction='none')
        loss_val.backward(gradient=torch.ones_like(loss_val))
        return output.grad.detach()

    def evaluate_model(self, test_loader, phase):
        """
        Evaluate the model on the given data and log and return the loss and top-1-accuracy.

        Args:
            test_loader: Data loader for the dataset to evaluate on.
            phase: Which data is evaluated, either "Train" or "Test". Used for the logged output.

        Returns:
            The computed loss and accuracy.
        """
        self.model.eval()

        total_loss, correct = 0, 0
        with torch.inference_mode():
            for _, data, label in test_loader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data, None)

                targets = self.prepare_targets(label)
                total_loss += self.loss(output, targets, reduction='sum').item()

                prediction = output.max(1, keepdim=True)[1]
                correct += prediction.eq(label.view_as(prediction)).sum().item()

        loss = total_loss / len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        logger.info(f"[{phase} Set] Loss: {loss:6f}, Accuracy: {accuracy:6.2%}")
        return loss, accuracy
