# Delayed Error Forward Projection: Feed-Forward-Only Training of Neural Networks

A PyTorch implementation comparing different approaches to feed-forward only training of neural networks.
It is based on the implementation of DRTP provided by Frenkel et al. at [https://github.com/ChFrenkel/DirectRandomTargetProjection](https://github.com/ChFrenkel/DirectRandomTargetProjection).

## Feed-Forward-Only Training of Neural Networks
The back-propagation algorithm used to efficiently compute the gradients when training neural networks has long been criticized for being biologically implausible as it relies on concepts that are not viable in the brain:
Two core issues are the weight transport and the update locking problem.
* Weight transport problem: In back-propagation, the forward weights are reused in the backward pass to propagate the gradients backward. This is biologically implausible as synapses are unidirectional and separate forward and backward pathways would require a synchronization of the weights between these pathways while the brain is inherently asynchronous and event-based.
* Update locking problem: When training with back-propagation, there are dependencies between the forward and backward passes. Having been processed in the forward pass, a layer needs to wait until all its downstream layers have been processed by both the forward and the backward pass before it can be updated. ´This can take a considerable amount of time and thus cause earlier layers to become desynchronized with the error.

Feed-forward-only training attempts to solve these problems by effectively removing the backward pass, thus training neural networks with only a forward pass.

## Installation and Usage

### Installation

This project can be installed as package using the provided `setup.py`, e.g. with  
```
pip install <path to project root>
```

### Usage

`feedforward/main.py` offers a command line interface to start the training.  
To get an overview of the available options, run 
```
python feedforward/main.py --help
```

For example, to train a single-layer fully-connected network with error-scaled DEFP on the synthetic dataset for 10 epochs, run
```
python feedforward/main.py --topology FC_500_FC_10 --algorithm Feed-Forward --dataset classification_synth --error-information delayed_error --epochs 10
```

### Tests

A simple test case is provided comparing the results after 10 epochs (test and train loss and top-1-accuracy) to results achieved with Frenkel's implementation.

For reproducible results on GPU with CUDA version ≥10.2, the environment needs to be configured as explained in [https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility), e.g. with  
```
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

To run the tests use
```
pytest <path to test sub-directory>
```  
Note that each test case trains a network for 10 epochs. Running all tests can thus require a significant amount of time, depending on the available hardware.
