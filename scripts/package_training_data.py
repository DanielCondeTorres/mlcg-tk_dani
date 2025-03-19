import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import RawDataset
from input_generator.utils import get_output_tag, CGFilesNotFound
from tqdm import tqdm
from time import ctime
import numpy as np
import pickle as pck
from typing import List, Union, Optional
from sklearn.model_selection import train_test_split
from jsonargparse import CLI
from copy import deepcopy
import h5py
import yaml


def package_training_data(
    dataset_name: str,
    names: List[str],
    dataset_tag: str,
    force_tag: str,
    training_data_dir: str,
    save_dir: str,
    save_h5: Optional[bool] = True,
    save_partition: Optional[bool] = True,
    single_protein: Optional[bool] = False,
    batch_size: int = 256,
    stride: int = 1,
    train_size: Optional[Union[float, int, None]] = 0.8,
    train_mols: Optional[List] = None,
    val_mols: Optional[List] = None,
    random_state: Optional[str] = None,
):
    """
    Computes structural features and accumulates statistics on dataset samples

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    dataset_tag : str
        Label given to all output files produced from dataset
    names : List[str]
        List of sample names
    force_tag : str
        Label given to produced delta forces and saved packaged data
    training_data_dir : str
        Path to directory from which input will be loaded
    save_dir : str
        Path to directory to which output will be saved
    save_h5 : bool
        Whether to save dataset h5 file(s)
    save_partition : bool
        Whether to save dataset partition file(s)
    single_protein : bool
        Whether the produced partition file should be for a single-molecule model
        Will be ignored if save_partition is False
    batch_size : int
        Number of samples of dataset to include in each training batch
    stride : int
        Integer by which to stride frames
    train_size : Union[float, int]
        Either the proportion (if float) or number of samples (if int) of molecules in training data
        If None, lists should be supplied for training and validation samples
    train_mols : Optional[List]
        Molecules to be used for training set
    val_mols : Optional[List]
        Molecules to be used for validation set
    random_state : Optional[str]
        Controls shuffling applied to the data before applying the split
    """

    dataset = RawDataset(dataset_name, names, dataset_tag)
    output_tag = get_output_tag([dataset_name, force_tag], placement="after")

    if save_h5:
        # Create H5 of training data
        fnout_h5 = osp.join(save_dir, f"{output_tag[1:]}.h5")

        with h5py.File(fnout_h5, "w") as f:
            metaset = f.create_group(dataset_name)
            for samples in tqdm(dataset, f"Packaging {dataset_name} dataset..."):
                try:
                    (
                        cg_coords,
                        cg_delta_forces,
                        cg_embeds,
                    ) = samples.load_training_inputs(
                        training_data_dir=training_data_dir,
                        force_tag=force_tag,
                    )
                except CGFilesNotFound as e:
                    print(
                        f"Sample {samples.name} has missing files - This entry will be skipped",
                        f", {e}",
                    )
                    continue

                name = f"{samples.tag}{samples.name}"
                hdf_group = metaset.create_group(name)

                hdf_group.create_dataset("cg_coords", data=cg_coords.astype(np.float32))
                hdf_group.create_dataset(
                    "cg_delta_forces", data=cg_delta_forces.astype(np.float32)
                )
                hdf_group.attrs["cg_embeds"] = cg_embeds
                hdf_group.attrs["N_frames"] = cg_coords.shape[0]

    if save_partition:
        # Create partition file
        fnout_part = osp.join(save_dir, f"partition{output_tag}.yaml")
        if single_protein:
            train_mols = [f"{dataset_tag}{name}" for name in names]
            val_mols = [f"{dataset_tag}{name}" for name in names]
            if train_size == None:
                raise ValueError(
                    "For single-protein partitions, a train size has to be specified"
                )
            if not isinstance(train_size, float):
                raise ValueError(
                    "For single-protein partitions, train_size has to be a float corresponding to the ratio of frames for training"
                )
            assert train_size <= 1.0, "train_size has to be a ratio of frames below 1.0"
        else:
            if train_mols == None and val_mols == None:
                if train_size == None:
                    raise ValueError(
                        "Either a train size or predefined lists for training and validation samples must be specified."
                    )

                train_mols, val_mols = train_test_split(
                    [f"{dataset_tag}{name}" for name in names],
                    train_size=train_size,
                    shuffle=True,
                    random_state=random_state,
                )
            elif train_mols != None:
                val_mols = deepcopy(names).remove(train_mols)
            elif val_mols != None:
                train_mols = deepcopy(names).remove(val_mols)

        partition_opts = {"train": {}, "val": {}}

        # make training data partition
        partition_opts["train"]["metasets"] = {}
        partition_opts["train"]["metasets"][dataset_name] = {
            "molecules": train_mols,
            "stride": stride,
        }
        partition_opts["train"]["batch_sizes"] = {dataset_name: batch_size}

        # make validation data partition
        partition_opts["val"]["metasets"] = {}
        partition_opts["val"]["metasets"][dataset_name] = {
            "molecules": val_mols,
            "stride": stride,
        }
        partition_opts["val"]["batch_sizes"] = {dataset_name: batch_size}

        if single_protein:
            partition_opts["train"]["metasets"][dataset_name]["detailed_indices"] = {
                "filename": "./splits"
            }
            for mol in train_mols:
                partition_opts["train"]["metasets"][dataset_name]["detailed_indices"][
                    mol
                ] = {
                    "seed": random_state,
                    "test_ratio": 0.0,
                    "val_ratio": 1.0 - train_size,
                }
        with open(fnout_part, "w") as ofile:
            yaml.dump(partition_opts, ofile)


def combine_datasets(
    dataset_names: List[str],
    save_dir: str,
    force_tag: Optional[str],
    save_h5: Optional[bool] = True,
    save_partition: Optional[bool] = True,
):
    """
    Computes structural features and accumulates statistics on dataset samples

    Parameters
    ----------
    dataset_names : List[str]
        List of dataset name to combine
    save_dir : str
        Path to directory from which datasets will be loaded and to which output will be saved
    force_tag : str
        Label given to produced delta forces and saved packaged data
    save_h5 : bool
        Whether to save dataset h5 file(s)
    save_partition : bool
        Whether to save dataset partition file(s)
    """

    datasets_label = "_".join(dataset_names)
    output_tag = get_output_tag([datasets_label, force_tag], placement="after")

    if save_h5:
        fnout_h5 = osp.join(save_dir, f"combined{output_tag}.h5")

        with h5py.File(fnout_h5, "w") as f:
            for dataset in dataset_names:
                f[dataset] = h5py.ExternalLink(
                    f"{dataset}{get_output_tag(force_tag, placement='after')}.h5",
                    f"/{dataset}",
                )

    if save_partition:
        fnout_part = osp.join(save_dir, f"partition{output_tag}.yaml")

        partition_opts = {"train": {}, "val": {}}
        partition_opts["train"]["metasets"] = {}
        partition_opts["train"]["batch_sizes"] = {}
        partition_opts["val"]["metasets"] = {}
        partition_opts["val"]["batch_sizes"] = {}

        for dataset in dataset_names:
            data_fn = osp.join(
                save_dir,
                f"partition_{dataset}{get_output_tag(force_tag, placement='after')}.yaml",
            )
            with open(data_fn, "r") as ifile:
                data_partition = yaml.safe_load(ifile)

            # make training data partition
            partition_opts["train"]["metasets"][dataset] = data_partition["train"][
                "metasets"
            ][dataset]
            partition_opts["train"]["batch_sizes"] = {
                dataset: data_partition["train"]["batch_sizes"]
            }

            # make validation data partition
            partition_opts["val"]["metasets"][dataset] = data_partition["val"][
                "metasets"
            ][dataset]
            partition_opts["val"]["batch_sizes"] = {
                dataset: data_partition["val"]["batch_sizes"]
            }

        with open(fnout_part, "w") as ofile:
            yaml.dump(partition_opts, ofile)


if __name__ == "__main__":
    print("Start package_training_data.py: {}".format(ctime()))

    CLI([package_training_data, combine_datasets])

    print("Finish package_training_data.py: {}".format(ctime()))
