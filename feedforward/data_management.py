import json
import logging
import pathlib
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch


class DataManager:
    """
    A class to manage loading and storing data with pickle or the corresponding methods provided by pytorch and pandas.
    Unless otherwise specified, they are stored by date and time in the following directory structure:
    <root_directory>/<YYYY-mm>/<YYYY-mm-dd--HH-MM-SS>[--suffix]
    """

    logger = logging.getLogger("DataManager")
    pickle_types = ["pandas", "torch", "python"]

    def __init__(self, data_directory=None, root_directory=None, suffix=None, pickle_protocol=pickle.DEFAULT_PROTOCOL):
        """
        Create a new data manager to load and store data as pickled files in the data_directory. If it does not exist,
        the data_directory and all missing parents are created on the first call to pickle.
        If no data_directory is specified, the directory is selected based on the current date and time as
        <root_directory>/<YYYY-mm>/<YYYY-mm-dd--HH-MM-SS>[--suffix]
        At least one of data_directory or root_directory should be defined.

        Args:
            data_directory: Path to the data directory to use. If it does not exist, the directory and all missing
            parents are created on the first call to pickle.
            root_directory: The root directory to use if no data_directory is specified.
            suffix: Optional suffix to append to the generated data_directory name if no specific directory is given.
            pickle_protocol: The pickle protocol to use in self.pickle(). See pickle documentation for details.
        """
        self.pickle_protocol = pickle_protocol
        if data_directory:
            self.data_directory = pathlib.Path(data_directory)
        else:
            self.data_directory = self.generate_data_directory_path(root_directory, suffix)
        self.logger.info(f"Using data directory {self.data_directory}")

    @staticmethod
    def generate_data_directory_path(root_directory, suffix=None):
        """
        Generate a data_directory path based on the current date and time as
        <root_directory>/<YYYY-mm>/<YYYY-mm-dd--HH-MM-SS>[--suffix]

        Args:
            root_directory: The root directory to attach the directory structure to.
            suffix: Optional suffix to append to directory name as '--{suffix}'.

        Returns:
            The generated path for the data_directory as pathlib.Path.
        """
        month = datetime.today().strftime('%Y-%m')
        now = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
        suffix = f"--{suffix}" if suffix else ""
        return pathlib.Path(root_directory) / month / (now + suffix)

    @staticmethod
    def determine_pickle_type(obj):
        """
        Determine the pickle_type of the given object, i.e. a string used to select the pickling method (pandas, torch
        or default python pickling). To find the pickle_type, the type() of the object is inspected and checked for the
        substrings "pandas" and "torch".

        Args:
            obj: The object to pickle.

        Returns:
            The pickle_type of the given object.
        """
        typename = str(type(obj))
        if "pandas" in typename:
            return "pandas"
        if "torch" in typename:
            return "torch"
        return "python"

    def pickle(self, obj, filename, overwrite=False):
        """
        Pickle the given object and save it to the file "<filename>.<pickle_type>.pickle" in the data directory.
        The pickle_type is determined with determine_pickle_type() and the object is pickled accordingly.
        If the data_directory does not yet exist, it and all its missing parents are created first.

        Args:
            obj: The object to pickle.
            filename: Base name of the file to write to. Will be appended with ".<pickle_type>.pickle".
            overwrite: Whether existing files should be overwritten. If False and the file already exists, a ValueError
            will be raised.

        Raises:
            ValueError: if the file already exists but overwrite is False.
        """
        self.data_directory.mkdir(parents=True, exist_ok=True)

        pickle_type = self.determine_pickle_type(obj)
        file_path = self.data_directory / f"{filename}.{pickle_type}.pickle"
        self.logger.debug(f"Pickling object to {file_path}")

        if not overwrite and file_path.exists():
            raise ValueError(f"Trying to pickle to {file_path} with overwrite={overwrite} but file already exists.")

        if pickle_type == "pandas":
            obj.to_pickle(file_path, protocol=self.pickle_protocol)
            return

        with open(file_path, 'wb') as file:
            if pickle_type == "torch":
                torch.save(obj, file, pickle_protocol=self.pickle_protocol)
            elif pickle_type == "python":
                pickle.dump(obj, file, protocol=self.pickle_protocol)

    def save_arrays(self, filename, overwrite=False, **named_arrays):
        """
        Store the given arrays in the specified file with numpy's savez.

        Args:
            filename: The file to store the arrays in. The suffix '.npz' will be appended.
            overwrite: Whether existing files should be overwritten. If False and the file already exists, a ValueError
            will be raised.
            **named_arrays: The arrays as named arguments. Each array is stored under the given name in the npz file.

        Raises:
            ValueError: if the file already exists but overwrite is False.
        """
        self.data_directory.mkdir(parents=True, exist_ok=True)

        file_path = self.data_directory / f"{filename}.npz"
        self.logger.debug(f"Saving arrays to {file_path}")

        if not overwrite and file_path.exists():
            raise ValueError(f"Trying to pickle to {file_path} with overwrite={overwrite} but file already exists.")

        np.savez(file_path, **named_arrays)

    def load_arrays(self, filename):
        """
        Load arrays stored with save_arrays from the given file with numpy's load.

        Args:
            filename: The file to load from.

        Returns:
            The loaded file as numpy NpzFile (a dict like data structure, arrays can be retrieved with
            npzFile[array_name].

        Raises:
             FileNotFoundError: if there is no such file in the data directory.
        """
        file_path = self.data_directory / filename
        if not file_path.exists():
            file_path = file_path.parent / (file_path.name + ".npz")

        if not file_path.exists():
            raise FileNotFoundError(f"No matching file for file name {filename} in {self.data_directory}.")

        self.logger.debug(f"Loading object from {file_path}")
        return np.load(file_path)

    def load(self, filename):
        """
        Load the object from the given file in the data directory.

        Args:
            filename: Base name of the file to read from. Will search for a file in the data directory matching
            "<filename>.<pickle_type>.pickle" where <pickle_type> is one of the types defined in
            DataManager.pickle_types.

        Returns:
            The object loaded from the file.

        Raises:
             FileNotFoundError: if there is no such file in the data directory.
        """
        matching_files = [(full_filename, re.search(fr"{re.escape(filename)}\.(.+?)\.pickle", str(full_filename)))
                          for full_filename in self.data_directory.glob(f"{filename}.*.pickle")]
        matching_files = [(full_filename, match.group(1)) for (full_filename, match) in matching_files
                          if match and match.group(1) in self.pickle_types]

        if not matching_files:
            raise FileNotFoundError(f"No matching file for base name {filename} in {self.data_directory}.")
        if len(matching_files) > 1:
            self.logger.warning(f"Multiple matching files for filename {filename} in {self.data_directory}, using first"
                                f": {[file for file, _ in matching_files]}")

        full_filename, pickle_type = matching_files[0]
        self.logger.debug(f"Loading object from {full_filename}")

        if pickle_type == "pandas":
            return pd.read_pickle(full_filename)

        with open(full_filename, 'rb') as file:
            if pickle_type == "torch":
                return torch.load(file)
            elif pickle_type == "python":
                return pickle.load(file)

    def save_json(self, obj, filename, overwrite=False):
        """
        Save the given object as json via json.dump() as <filename>.json in the data directory.
        If the data_directory does not yet exist, it and all its missing parents are created first.

        Args:
            obj: The object to save as json.
            filename: Base name of the file to write to. Will be appended with ".json".
            overwrite: Whether existing files should be overwritten. If False and the file already exists, a ValueError

        Raises:
            ValueError: if the file already exists but overwrite is False.
        """
        self.data_directory.mkdir(parents=True, exist_ok=True)

        file_path = self.data_directory / f"{filename}.json"
        self.logger.debug(f"Saving json to {file_path}")

        if not overwrite and file_path.exists():
            raise ValueError(f"Trying to pickle to {file_path} with overwrite={overwrite} but file already exists.")

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(obj, file, ensure_ascii=False, indent=4)

    def load_json(self, filename):
        """
        Load the data from the given json file at <data directory>/<filename>{.json}.

        Args:
            filename: The name of the file to load json data from. The '.json' file extension is optional.

        Returns:
            The result of json.load.

        Raises:
             FileNotFoundError: if there is no such file in the data directory.
        """
        file_path = self.data_directory / filename
        if not file_path.exists():
            file_path = file_path.parent / (file_path.name + ".json")

        if not file_path.exists():
            raise FileNotFoundError(f"No matching file for file name {filename} in {self.data_directory}.")

        self.logger.debug(f"Loading object from {file_path}")
        with open(file_path, 'r') as file:
            return json.load(file)
