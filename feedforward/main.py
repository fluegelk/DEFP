import argparse
import collections
import logging
import pathlib
import sys
import uuid

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

import feedforward


def merge_dicts_resolve_collisions(**named_dicts):
    """
    Merge a collection of named dictionaries. Resolve collisions by prepending the dictionary name, i.e.
    '<dict name>__<original key>' to duplicate keys.

    Args:
        **named_dicts: Collection of named dictionaries to merge.

    Returns:
        The merged dictionary containing all key-value pairs of the input dictionaries, but collisions are resolved
        by renaming the duplicate keys.
    """
    key_frequencies = collections.Counter([key for dictionary in named_dicts.values() for key in dictionary])
    duplicate_keys = set(key for key, frequency in key_frequencies.items() if frequency > 1)

    return {
        f'{name}__{key}' if key in duplicate_keys else key: value
        for name, content in named_dicts.items() for key, value in content.items()
    }


def report_memory_stats(device, prefix=''):
    """
    Report the current cuda peak memory statistics (allocated and reserved) on the given device and log them on INFO log
    level. Does nothing if the given device is no cuda device.

    Args:
        device: The torch device to analyze.
        prefix: Optional prefix to prepend to the logging output.
    """
    if device.type != 'cuda':
        return
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)

    logging.info(f"{prefix}[Memory Stats]: {max_memory_allocated} (allocated) | {max_memory_reserved} (reserved)")


def run_training(seed, epochs, algorithm, train_mode, device, loss_name, optimizer_name, lr_scheduler_name, initial_lr,
                 data_kwargs, model_kwargs, training_kwargs):
    # device: 'cpu' or 'cuda'
    # data_kwargs should contain: dataset_name, batch_size, test_batch_size, [shuffle_train_set]
    # topology_string: string describing the network topology as expected by TopologyParser
    # model_kwargs should contain: dropout, convolution_activation, hidden_activation, output_activation, ...

    torch.manual_seed(seed)

    torch_device = torch.device(device)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(torch_device)
    report_memory_stats(torch_device, 'After reset ')
    data_loader_kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}

    train_loader, train_test_loader, test_loader, input_size, number_of_classes = feedforward.data.prepare_data(
        **data_kwargs, data_loader_kwargs=data_loader_kwargs)
    report_memory_stats(torch_device, 'After data prep ')

    model = feedforward.model.create_model(algorithm, input_size=input_size, **model_kwargs)
    model.to(torch_device)
    report_memory_stats(torch_device, 'After model creation ')

    loss = feedforward.training.create_loss_function(loss_name)
    optimizer = feedforward.training.create_optimizer(optimizer_name, model, lr=initial_lr)
    lr_scheduler = feedforward.training.create_learning_rate_scheduler(
        lr_scheduler_name, optimizer) if lr_scheduler_name else None

    training = feedforward.training.Training(model, loss, optimizer, lr_scheduler, torch_device, train_loader,
                                             train_test_loader, test_loader, number_of_classes=number_of_classes,
                                             **training_kwargs)
    report_memory_stats(torch_device, 'After training setup ')
    training.train(epochs, algorithm)

    config_dict = merge_dicts_resolve_collisions(
        data=data_kwargs, model=model_kwargs, training=training_kwargs, general={
            'seed': seed, 'epochs': epochs, 'algorithm': algorithm, 'train_mode': train_mode, 'device': device,
            'loss_name': loss_name, 'optimizer_name': optimizer_name, 'lr_scheduler_name': lr_scheduler_name,
            'initial_lr': initial_lr})
    training.results.add_constant_columns(config_dict)
    results = training.results.build_dataframe()

    return model, results, config_dict


def save_results(data_manager, file_prefix='', **named_results):
    """
    Save the given results with the provided data manager. Results with name 'config' are saved as json, all others
    using pickle. Errors are logged but ignored.

    Args:
        data_manager: The DataManager object to save the results with.
        file_prefix: Optional file prefix to prepend to the file names.
        **named_results: Name-data pairs of results to save.
    """
    jsons = ['config']
    for name, result in named_results.items():
        file_name = f'{file_prefix}{name}'
        try:
            if name in jsons:
                data_manager.save_json(result, file_name)
            else:
                data_manager.pickle(result, file_name)
        except Exception as e:
            logging.error(f'Error saving {name}: {e}')


def main(outpath, trials, epochs, algorithm, train_mode, loss, optimizer, lr_scheduler, lr, dataset,
         batch_size, test_batch_size, topology, dropout, conv_act, hidden_act,
         output_act, shuffle_train_set=None, feedback_weight_initialization=None, ff_implementation=None,
         error_information=None, verbose=None, cpu=False, **unused):
    device, data_kwargs, model_kwargs, training_kwargs = prepare_parameters(
        dataset, batch_size, test_batch_size, topology, dropout, conv_act, hidden_act, output_act, shuffle_train_set,
        feedback_weight_initialization, ff_implementation, error_information, verbose, cpu, **unused)

    directory_suffix = f"{uuid.uuid4()}--{dataset}_{train_mode}"
    data_manager = feedforward.data_management.DataManager(root_directory=outpath,
                                                           suffix=directory_suffix) if outpath else None

    for seed in range(trials):
        model, results, config_dict = run_training(seed, epochs, algorithm, train_mode, device, loss, optimizer,
                                                   lr_scheduler, lr, data_kwargs, model_kwargs, training_kwargs)
        if data_manager:
            save_results(data_manager, file_prefix=f'{seed}_', model=model.state_dict(), results=results, config=config_dict)


def prepare_parameters(dataset, batch_size, test_batch_size, topology, dropout, conv_act, hidden_act, output_act,
                       shuffle_train_set=None, feedback_weight_initialization=None, implementation=None,
                       error_information=None, verbose=None, cpu=False, **unused):
    if unused:
        logging.info(f'Unused parameters passed to prepare_parameters: {unused}')

    device = 'cpu' if cpu or not torch.cuda.is_available() else 'cuda'

    def remove_none(dictionary):
        return {key: value for key, value in dictionary.items() if value is not None}

    data_kwargs = {'dataset_name': dataset, 'batch_size': batch_size, 'test_batch_size': test_batch_size,
                   'shuffle_train_set': shuffle_train_set}
    data_kwargs = remove_none(data_kwargs)

    model_kwargs = {
        'topology': topology,
        'dropout': dropout,
        'convolution_activation': conv_act,
        'hidden_activation': hidden_act,
        'output_activation': output_act,
        'feedback_weight_initialization': feedback_weight_initialization,
        'implementation': implementation
    }
    model_kwargs = remove_none(model_kwargs)

    training_kwargs = {'error_information': error_information, 'verbose': verbose}
    training_kwargs = remove_none(training_kwargs)

    return device, data_kwargs, model_kwargs, training_kwargs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, default=None, help='Where to store the outputs.')

    parser.add_argument('--dataset', type=str, choices=['regression_synth', 'classification_synth', 'MNIST', 'CIFAR10',
                                                        'CIFAR10aug'], required=True, help='The dataset to train on.')
    topology_help = 'The network topology of the model to train. Layers are separated by \'_\' and described as' \
                    'FC_{output units} for fully-connected layers and ' \
                    'CONV_{output channels}_{kernel size}_{stride}_{padding} for convolutional layers.'
    parser.add_argument('--topology', type=str, required=True, help=topology_help)

    parser.add_argument('--algorithm', choices=['BP', 'Feed-Forward'], required=True,
                        help='The training algorithm. BP for back-propagation, Feed-Forward for a '
                             'feed-forward training approach.')
    parser.add_argument('--feedback-weight-initialization', type=str,
                        help="A torch init function, e.g. 'kaiming_uniform' for torch.nn.init.kaiming_uniform_.")
    parser.add_argument('--ff-implementation', type=str, choices=['true_feed_forward', 'gradient_replacement'],
                        default='true_feed_forward',
                        help="Switch between the new (true_feed_forward) and old (gradient_replacement) implementation"
                             "for feed forward training.")
    parser.add_argument(
        '--error-information', choices=['targets', 'delayed_error', 'delayed_loss'],
        help="Error information to use with the feed-forward training. 'targets' corresponds to DRTP, while "
             "'delayed_error' and 'delayed_loss' correspond to DEFP with the corresponding delayed information.",
        default='targets')

    parser.add_argument('--optimizer', choices=['SGD', 'NAG', 'Adam', 'RMSprop'], help='The optimizer to train with.')
    parser.add_argument('--lr', type=float, help='The initial learning rate.')
    parser.add_argument('--lr-scheduler', help='Optional learning rate scheduler, '
                                               'parsed by training.create_learning_rate_scheduler.', type=str)

    parser.add_argument('--trials', type=int, default=1, help='The number of independent training runs.')
    parser.add_argument('--epochs', type=int, help='The number of epochs to train.')

    args = parser.parse_args()

    args.train_mode = args.algorithm
    if args.algorithm == 'Feed-Forward':
        train_modes = {'targets': 'DRTP', 'delayed_error': 'error-scaled', 'delayed_loss': 'loss-scaled'}
        args.train_mode = train_modes[args.error_information]

    logging.info(vars(args))
    args_with_defaults = feedforward.parameters.merge_with_default_parameters(**vars(args))
    return {key.replace('-', '_'): value for key, value in args_with_defaults.items()}


if __name__ == '__main__':
    logging.basicConfig(level='INFO', format="%(asctime)s\t\t%(levelname)-8s %(name)-20s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    arguments = parse_arguments()
    logging.info(arguments)
    main(**arguments)
