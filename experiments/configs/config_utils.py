import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

def read_config(config_file):
    """Read a config file in YAML.
    Parameters
    ----------
    config_file : str
        Path towards the con fig file in YAML.
    Returns
    -------
    dict
        The parsed config
    Raises
    ------
    FileNotFoundError
        If the config file does not exist
    """
    if not (os.path.exists(config_file)):
        raise FileNotFoundError("Could not find the config to read.")
    with open(config_file, "r") as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)
    return dict


def get_config_file_path(dataset_name, debug):
    """Get the config_file path in real or debug mode.
    Parameters
    ----------
    dataset_name: str
        The name of the dataset to get the config from.
    debug : bool
       The mode in which we download the dataset.
    Returns
    -------
    str
        The path towards the config file.
    """
    assert dataset_name in [
        "fed_heart_disease",
        "fed_mnist",
        "fed_cifar10",
        "fed_snli"
    ], f"Dataset name {dataset_name} not valid."
    config_file_name = (
        f"{dataset_name}_debug.yaml" if debug else f"{dataset_name}.yaml"
    )
    path_to_config_file_folder = str(Path(os.path.realpath(__file__)).parent.resolve())
    config_file = os.path.join(path_to_config_file_folder, config_file_name)
    return config_file


def create_config(output_folder, debug, dataset_name="fed_camelyon16"):
    """Create or modify config file by writing the absolute path of \
        output_folder in its dataset_path key.
    Parameters
    ----------
    output_folder : str
        The folder where the dataset will be downloaded.
    debug : bool
        Whether or not we are in debug mode.
    dataset_name: str
        The name of the dataset to get the config from.
    Returns
    -------
    Tuple(dict, str)
        The parsed config and the path to the file written on disk.
    Raises
    ------
    ValueError
        If output_folder is not a directory.
    """
    if not (os.path.isdir(output_folder)):
        raise ValueError(f"{output_folder} is not recognized as a folder")

    config_file = get_config_file_path(dataset_name, debug)

    if not (os.path.exists(config_file)):
        dataset_path = os.path.realpath(output_folder)
        dict = {
            "dataset_path": dataset_path,
            "download_complete": False,
            "preprocessing_complete": False,
        }

        with open(config_file, "w") as file:
            yaml.dump(dict, file)
    else:
        dict = read_config(config_file)

    return dict, config_file


def write_value_in_config(config_file, key, value):
    """Update config_file by modifying one of its key with its new value.
    Parameters
    ----------
    config_file : str
        Path towards a config file
    key : str
        A key belonging to download_complete, preprocessing_complete, dataset_path
    value : Union[bool, str]
        The value to write for the key field.
    Raises
    ------
    ValueError
        If the config file does not exist.
    """
    if not (os.path.exists(config_file)):
        raise FileNotFoundError(
            "The config file doesn't exist. \
            Please create the config file before updating it."
        )
    dict = read_config(config_file)
    dict[key] = value
    with open(config_file, "w") as file:
        yaml.dump(dict, file)


def check_dataset_from_config(dataset_name, debug):
    """Verify that the dataset is ready to be used by reading info from the config
    files.
    Parameters
    ----------
    dataset_name: str
        The name of the dataset to check
    debug : bool
        Whether to use the debug dataset or not.
    Returns
    -------
    dict
        The parsed config.
    Raises
    ------
    ValueError
        The dataset download or preprocessing did not finish.
    """
    try:
        dict = read_config(get_config_file_path(dataset_name, debug))
    except FileNotFoundError:
        if debug:
            raise ValueError(
                f"The dataset was not downloaded, config file "
                "not found for debug mode. Please refer to "
                "the download instructions inside "
                f"FLamby/flamby/datasets/{dataset_name}/README.md"
            )
        else:
            debug = True
            print(
                "WARNING USING DEBUG MODE DATASET EVEN THOUGH DEBUG WAS "
                "SET TO FALSE, COULD NOT FIND NON DEBUG DATASET CONFIG FILE"
            )
            try:
                dict = read_config(get_config_file_path(dataset_name, debug))
            except FileNotFoundError:
                raise ValueError(
                    f"It seems the dataset {dataset_name} was not downloaded as "
                    "the config file is not found for either normal or debug "
                    "mode. Please refer to the download instructions inside "
                    f"FLamby/flamby/datasets/{dataset_name}/README.md"
                )
    if not (dict["download_complete"]):
        raise ValueError(
            f"It seems the dataset {dataset_name} was only partially downloaded"
            "please restart the download script to finish the download."
        )
    if not (dict["preprocessing_complete"]):
        raise ValueError(
            f"It seems the preprocessing for dataset {dataset_name} is not "
            "yet finished please run the appropriate preprocessing scripts "
            "before use"
        )
    return dict


def accept_license(license_link, dataset_name, save_agreement=True):
    """This function forces the user to accept the license terms before
    proceeding with the download.

    Parameters
    ----------
    license_link : str
        The link towards the data terms of the original dataset.
    dataset_name: str
        The name of the dataset associated with the license
    save_agreement: bool
        Whether or not to save a file if one already answered yes.
    """
    assert dataset_name in [
        "fed_camelyon16",
        "fed_heart_disease",
        "fed_isic2019",
        "fed_lidc_idri",
        "fed_ixi",
        "fed_kits19",
        "fed_tcga_brca",
    ], f"Dataset name {dataset_name} not valid."

    datasets_dir = str(Path(os.path.realpath(datasets.__file__)).parent.resolve())
    path_to_dataset_folder = os.path.join(
        datasets_dir, dataset_name, "dataset_creation_scripts"
    )
    license_acceptance_file_path = os.path.join(
        path_to_dataset_folder, f"license_agreement_{dataset_name}"
    )

    # If the license acceptance file is found we do nothing
    if os.path.exists(license_acceptance_file_path):
        return

    while True:
        answer = input(
            "Have you taken the time to read and accept the data terms on the"
            " original website, available at the following link: "
            f"{license_link} ? | (y/n)\n\n\n"
        )
        if any(answer.lower() == f for f in ["yes", "y", "1", "ye"]):
            print("Saving license agreement")
            Path(license_acceptance_file_path).touch()
            print("You may now proceed to download.\n")
            break

        elif any(answer.lower() == f for f in ["no", "n", "0"]):
            print(
                "Since you have not read and accepted the license terms the "
                "download of the dataset is aborted. Please come back when you"
                " have fulfilled this legal obligation."
            )
            sys.exit()
        else:
            print(
                "If you wish to proceed with the download you need to read and"
                " accept the license and data terms of the original data owners."
                " Please read and accept the terms and answer yes.\n\n\n"
            )


def seaborn_styling(figsize=(40, 20), legend_fontsize=24, labelsize=24):
    """This is used to set homogeneous default params for seaborn.
    Parameters
    ----------
    figsize : tuple
        The default figure size in inches.
    legend_fontsize: int
        Default fontsize of the legends labels.
    labelsize: int
        Default size of labels.
    """
    sns.set_theme()
    sns.set_style("darkgrid")
    plt.grid()
    figure = {"figsize": figsize}
    axes = {"labelsize": labelsize}
    matplotlib.rc("axes", **axes)
    matplotlib.rc("figure", **figure)
    plt.rcParams["savefig.dpi"] = 300
    plt.rc("legend", fontsize=legend_fontsize)
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
