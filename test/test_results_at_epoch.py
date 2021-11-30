import logging
import pandas as pd
import pathlib
import pytest
import sys
import torch

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feedforward import main, parameters


class TestResultsAtEpoch10:
    """
    Compare the results of the current implementation to the results using the pytorch implementation of DRTP by
    Frenkel et al. after 10 training epochs.

    IMPORTANT: for reproducible results on GPU with CUDA version ≥10.2 set environment variables as explained in
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    e.g. with export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    """

    EPOCH = 10  # the number of epochs to train
    RANDOM_SEED = 0  # the random seed to use

    # the results using the implementation by Frenkel et al. available at
    # https://github.com/ChFrenkel/DirectRandomTargetProjection
    EXPECTED_RESULTS = pd.DataFrame([
        ["classification_synth", "FC_256_FC_10", "BP", 0.91, 0.91, 2.72, 2.79],
        ["classification_synth", "FC_256_FC_10", "DRTP", 0.94, 0.94, 1.61, 1.73],

        ["MNIST", "FC1-500", "BP", 0.97, 0.97, 0.22, 0.24],
        ["MNIST", "FC1-500", "DRTP", 0.93, 0.93, 0.42, 0.42],
        ["MNIST", "CONV_MNIST", "BP", 1.00, 0.99, 0.01, 0.10],
        ["MNIST", "CONV_MNIST", "DRTP", 0.97, 0.97, 0.17, 0.19],

        ["CIFAR10", "FC1-500", "BP", 0.44, 0.42, 3.17, 3.25],
        ["CIFAR10", "FC1-500", "DRTP", 0.46, 0.43, 2.85, 2.95],
    ], columns=["dataset", "topology", "train_mode", "train_accuracy", "test_accuracy", "train_loss", "test_loss"])

    def get_expected(self, dataset, topology, train_mode):
        """
        Get the expected results for the given dataset, network topology and training mode.

        Args:
            dataset: The dataset.
            topology: The network topology.
            train_mode: The training mode ('BP' or 'DRTP').

        Returns:
            The row(s) from the expected results table matching the given dataset, network topology and training mode.
            Empty if there is no matching entry.
        """
        data = self.EXPECTED_RESULTS
        return data[(data.dataset == dataset) & (data.topology == topology) & (data.train_mode == train_mode)]

    @staticmethod
    def prepare_parameters(dataset, topology, train_mode, **kwargs):
        """
        Prepare the parameters for main.run_training based on the given dataset, network topology and training mode, and
        any additional parameters.

        Args:
            dataset: The dataset.
            topology: The network topology.
            train_mode: The training mode ('BP' or 'DRTP').
            **kwargs: Optional additional parameters to be passed to merge_with_default_parameters.

        Returns:
            merged_parameters: Dictionary containing the whole configuration.
            device: The device to use as string, 'cpu' or 'cuda'.
            data_kwargs: The data_kwargs to be passed to main.run_training.
            model_kwargs: The model_kwargs to be passed to main.run_training.
            training_kwargs: The training_kwargs to be passed to main.run_training.
        """
        algorithm = train_mode if train_mode == 'BP' else 'Feed-Forward'
        specified_parameters = {'dataset': dataset, 'topology': topology, 'train_mode': train_mode,
                                'algorithm': algorithm, 'error_information': 'targets', **kwargs}
        merged_parameters = parameters.merge_with_default_parameters(**specified_parameters)
        merged_parameters = {key.replace('-', '_'): value for key, value in merged_parameters.items()}

        device, data_kwargs, model_kwargs, training_kwargs = main.prepare_parameters(**merged_parameters)
        return merged_parameters, device, data_kwargs, model_kwargs, training_kwargs

    def run_test(self, dataset, topology, train_mode, implementation='true_feed_forward'):
        """
        Run the test for the given given dataset, network topology, training mode, and implementation.
        Train the model for 10 epochs and compare the resulting train and test accuracy and loss to the expected values.

        Args:
            dataset: The dataset.
            topology: The network topology.
            train_mode: The training mode ('BP' or 'DRTP').
            implementation: The implementation for feed-forward training: 'true_feed_forward' or 'gradient_replacement'.
        """
        torch.use_deterministic_algorithms(True)
        config, device, data_kwargs, model_kwargs, training_kwargs = self.prepare_parameters(
            dataset, topology, train_mode, epochs=self.EPOCH, verbose=2, implementation=implementation)
        logging.info(f"Configuration: {config}")
        model, results, _ = main.run_training(
            self.RANDOM_SEED, self.EPOCH, config['algorithm'], config['train_mode'], device, config['loss'],
            config['optimizer'], None, config['lr'], data_kwargs, model_kwargs, training_kwargs)

        actual = results[results.epoch == self.EPOCH][['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss']]
        train_accuracy, test_accuracy, train_loss, test_loss = actual.iloc[0]

        expected = self.get_expected(dataset, topology, train_mode)

        assert pytest.approx(train_accuracy, abs=0.1) == expected.train_accuracy.item()
        assert pytest.approx(test_accuracy, abs=0.1) == expected.test_accuracy.item()
        assert pytest.approx(train_loss, abs=0.1) == expected.train_loss.item()
        assert pytest.approx(test_loss, abs=0.1) == expected.test_loss.item()

    def test_classification_synth_bp_results(self):
        self.run_test("classification_synth", "FC_256_FC_10", "BP")

    def test_classification_synth_drtp_results(self):
        self.run_test("classification_synth", "FC_256_FC_10", "DRTP")

    def test_classification_synth_drtp_results_old_implementation(self):
        self.run_test("classification_synth", "FC_256_FC_10", "DRTP", "gradient_replacement")

    def test_mnist_bp_fc_results(self):
        self.run_test("MNIST", "FC1-500", "BP")

    def test_mnist_drtp_fc_results(self):
        self.run_test("MNIST", "FC1-500", "DRTP")

    def test_mnist_bp_conv_results(self):
        self.run_test("MNIST", "CONV_MNIST", "BP")

    def test_mnist_drtp_conv_results(self):
        self.run_test("MNIST", "CONV_MNIST", "DRTP")

    def test_cifar_bp_results(self):
        self.run_test("CIFAR10", "FC1-500", "BP")

    def test_cifar_drtc_results(self):
        self.run_test("CIFAR10", "FC1-500", "DRTP")
