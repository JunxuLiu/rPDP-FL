# Heart Disease

The Heart Disease dataset [1] was collected in 1988 in four centers:
Cleveland, Hungary, Switzerland and Long Beach V. We do not own the
copyright of the data: everyone using this dataset should abide by its
licence and give proper attribution to the original authors. It is
available for download
[here](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Dataset description
Please refer to the [dataset website](https://archive.ics.uci.edu/dataset/45/heart+disease)
for an exhaustive data sheet. The table below provides a high-level description
of the dataset.

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Heart Disease dataset.
| Dataset size       | 39,6 KB.
| Centers            | 4 centers - Cleveland, Hungary, Switzerland and Long Beach V.
| Records per center | Train/Test: 199/104, 172/89, 30/16, 85/45.
| Inputs shape       | 16 features (tabular data).
| Total nb of points | 740.
| Task               | Binary classification

### License and data usage terms
This dataset is licensed under a Creative Commons Attribution
4.0 International (**CC-BY 4.0**) license by its authors.
*Anyone using this dataset should abide by its*
*licence and give proper attribution to the original authors.*

### Ethics
As per the [dataset website](https://archive.ics.uci.edu/dataset/45/heart+disease),
sensitive entries of the dataset were removed by the original authors:

> The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

## Download and preprocessing instructions

To download the data,
First cd into the `dataset_creation_scripts` folder, then simply run the following command:
```
python download.py --output-folder /your/path/towards/dataset
```
This will download 38.6ko of data.

**IMPORTANT :** If you choose to relocate the dataset after downloading it, it is
imperative that you run the following script otherwise all subsequent scripts will not find it:
```
python update_config.py --new-path /new/path/towards/dataset
```

## Using the dataset

Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from datasets.fed_heart_disease import FedHeartDisease, HeartDiseaseRaw

# To load the full dataset as a pytorch dataset
rawdata = HeartDiseaseRaw(data_path=data_path)
# To load the first center as a pytorch dataset
center0 = FedHeartDisease(rawdata=rawdata, center=0, train=True)
# To load the second center as a pytorch dataset
center1 = FedHeartDisease(rawdata=rawdata, center=1, train=True)
# To sample batches from each of the local datasets use the traditional pytorch API
from torch.utils.data import DataLoader as dl

X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0)).next()
```
More informations on how to train model and handle flamby datasets in general are available in the [FLamby](https://github.com/owkin/FLamby) repository.

## References

[1] Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, Detrano,
Robert & M.D., M.D.. (1988). Heart Disease. UCI Machine Learning
Repository.

## Citing FLamby

@inproceedings{NEURIPS2022_232eee8e,
 author = {Ogier du Terrail, Jean and Ayed, Samy-Safwan and Cyffers, Edwige and Grimberg, Felix and He, Chaoyang and Loeb, Regis and Mangold, Paul and Marchand, Tanguy and Marfoq, Othmane and Mushtaq, Erum and Muzellec, Boris and Philippenko, Constantin and Silva, Santiago and Tele\'{n}czuk, Maria and Albarqouni, Shadi and Avestimehr, Salman and Bellet, Aur\'{e}lien and Dieuleveut, Aymeric and Jaggi, Martin and Karimireddy, Sai Praneeth and Lorenzi, Marco and Neglia, Giovanni and Tommasi, Marc and Andreux, Mathieu},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {5315--5334},
 publisher = {Curran Associates, Inc.},
 title = {FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/232eee8ef411a0a316efa298d7be3c2b-Paper-Datasets_and_Benchmarks.pdf},
 volume = {35},
 year = {2022}
}
