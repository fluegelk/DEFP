import logging
import pandas as pd

logger = logging.getLogger('DEFP')

default_parameters = {
    'cpu': False,
    'dataset': 'MNIST',
    'train_mode': 'DRTP',
    'optimizer': 'NAG',
    'loss': 'BCE',
    'freeze_conv_layers': False,
    'fc_zero_init': False,
    'dropout': 0,
    'trials': 1,
    'epochs': 100,
    'batch_size': 100,
    'test_batch_size': 1000,
    'lr': 1e-4,
    'topology': 'CONV_32_5_1_2_FC_1000_FC_10',
    'conv_act': 'tanh',
    'hidden_act': 'tanh',
    'output_act': 'sigmoid',
}

classification_synth_config = {
    "dataset": "classification_synth",
    "epochs": 500,
    "topology": "FC_256_FC_500_FC_500_FC_10",
    "hidden_act": "tanh",
    "output_act": "sigmoid",
    "loss": "BCE",
    "lr": 5e-4,
    "batch_size": 50,
    "trials": 10,
}

# Learning rates as given in Table 3 in DRTP by Frenkel et al.
learning_rates = pd.DataFrame([
    ["MNIST", "FC1", 1.5e-4, 5e-4, 1.5e-4, 1.5e-4],
    ["MNIST", "FC2", 5e-4, 5e-4, 1.5e-4, 1.5e-4],
    ["MNIST", "CONV (random)", 5e-5, 5e-4, 5e-4, 5e-4],
    ["MNIST", "CONV (trained)", 5e-4, 1.5e-4, 1.5e-4, 1.5e-4],

    ["CIFAR10", "FC1", 1.5e-5, 5e-5, 1.5e-4, 1.5e-4],
    ["CIFAR10", "FC2", 5e-6, 5e-5, 5e-5, 5e-5],
    ["CIFAR10", "CONV (random)", 5e-6, 1.5e-4, 1.5e-4, 1.5e-4],
    ["CIFAR10", "CONV (trained)", 1.5e-4, 1.5e-5, 5e-5, 5e-5],
], columns=["dataset", "topology", "BP", "sDFA", "DRTP", "DRTP+"])


def topology_mapper(topology):
    """Map keys from topologies to the topology column of the learning_rates data frame."""
    if topology.startswith("FC1"):
        return "FC1"
    elif topology.startswith("FC2"):
        return "FC2"
    elif topology.startswith("CONV"):
        return "CONV (trained)"
    raise ValueError(f"Invalid topology name {topology}")


def get_learning_rate(dataset, topology, algorithm):
    """Get the correct learning rate (as specified in Table 3) for a given dataset, topology and algorithm."""
    row = learning_rates[(learning_rates.dataset == dataset) & (learning_rates.topology == topology_mapper(topology))]
    return row[algorithm].iloc[0]


topologies = {
    "FC1-500": "FC_500_FC_10",
    "FC1-1000": "FC_1000_FC_10",
    "FC2-500": "FC_500_FC_500_FC_10",
    "FC2-1000": "FC_1000_FC_1000_FC_10",
    "CONV_MNIST": "CONV_32_5_1_2_FC_1000_FC_10",
    "CONV_CIFAR10": "CONV_64_3_1_1_CONV_256_3_1_1_FC_1000_FC_1000_FC_10",
}
topology_to_name = {description: name for name, description in topologies.items()}


def get_topology_name_and_description(topology):
    """
    Try to convert a given topology parameter to a pair of name and description.
    :param topology: Can be either the name of a known topology or a topology description string.
    :return: The topology name (or None if no name is known) and the corresponding topology description.
    """
    if topology in topologies:  # passed topology is a topology name
        return topology, topologies[topology]
    elif topology in topology_to_name:  # passed topology is a named topology description
        return topology_to_name[topology], topology
    else:  # passed topology is neither name nor description of a known topology. Assume it is a topology description.
        return None, topology


base_config_image_classification = {
    "hidden_act": "tanh",
    "output_act": "sigmoid",
    "loss": "BCE",
    "optimizer": "Adam",
    "trials": 10
}

base_config_mnist = {
    **base_config_image_classification,
    "dataset": "MNIST",
    "epochs": 100,
    "batch_size": 60,
}

base_config_cifar10 = {
    **base_config_image_classification,
    "dataset": "CIFAR10",
    "epochs": 200,
    "batch_size": 100,
}


def merge_with_default_parameters(**kwargs):
    logger.debug(f'Specified parameters: {kwargs}')

    def get_with_default(key, dictionary, defaults):
        return dictionary.get(key, defaults[key])

    dataset = get_with_default('dataset', kwargs, default_parameters)
    not_none_kwargs = {key: value for key, value in kwargs.items() if value is not None}

    if dataset == "classification_synth":
        merged_parameters = {**kwargs, **default_parameters, **classification_synth_config, **not_none_kwargs}
    elif dataset in ["MNIST", "CIFAR10"]:
        base_parameters = {**default_parameters, **(base_config_mnist if dataset == "MNIST" else base_config_cifar10)}

        topology = get_with_default('topology', kwargs, base_parameters)
        topology_name, topology_description = get_topology_name_and_description(topology)

        if 'lr' not in kwargs:  # if learning rate is not given, try to infer it from the other parameters
            train_mode = get_with_default('train_mode', kwargs, base_parameters)
            try:
                not_none_kwargs['lr'] = get_learning_rate(dataset, topology_name, train_mode)
            except KeyError:
                pass
        merged_parameters = {**kwargs, **base_parameters, **not_none_kwargs, 'topology': topology_description}
    else:
        merged_parameters = {**kwargs, **default_parameters, **not_none_kwargs}
    logger.debug(f'Parameters merged with defaults: {merged_parameters}')
    return merged_parameters
