<p align="center">
  <a href="https://praig.ua.es/"><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">Few-Shot Symbol Classification via Self-Supervised Learning and Nearest Neighbor</h1>

<h4 align="center">Full text available <a href="https://doi.org/10.1016/j.patrec.2023.01.014" target="_blank">here</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>


## About

## Load data example:

If you want to replicate just the load of the data, check "calculate_distances" function in "models/FewShotModels.py"

In that function, all the datasets are loaded without partition sets. 

```
  ## Load only the training src dataset
  pretrained_sources = True if Constants["NoSrcDataset"] else pretrained_sources

  # If no validation src paramenter, Xval is empty
  data_dict = load_supervised_data(ds_name=ds_name, min_occurence=50, all_datasets=False, pretrained_sources=pretrained_sources, boots_iter=0)
  
  
  XTrain, YTrain = data_dict["X_tgt"], data_dict["Y_tgt"]
```
"XTrain" and "YTrain" correspond to the whole loaded dataset. 


## How To Use

To execute an experiment, please execute this line

```python3 ./scripts/auto_paralel_exps.sh <experiment_number> <simultaneous_executions>
```

There are also a few relevant scripts. 

#### Extract the logs into simpler tables:

```python3 logs_csv/filter_logs_csv.py
```

#### Compare datasets by predictions and generated embeddings:

```python3 scripts/complementary_comp/main_complementary_comparison_methods.py
```




## Citations

```bibtex
@article{alfaro2023few,
  title     = {{Few-Shot Symbol Classification via Self-Supervised Learning and Nearest Neighbor}},
  author    = {Alfaro-Contreras, Mar{\'\i}a and R{\'\i}os-Vila, Antonio and Valero-Mas, Jose J and Calvo-Zaragoza, Jorge},
  journal   = {{Pattern Recognition Letters}},
  volume    = {167},
  pages     = {1--8},
  year      = {2023},
  publisher = {Elsevier},
  doi       = {10.1016/j.patrec.2023.01.014},
}

@inproceedings{rios2022few,
  title     =   {{Few-Shot Music Symbol Classification via Self-Supervised Learning and Nearest Neighbor}},
  author    =   {R{\'\i}os-Vila, Antonio and Alfaro-Contreras, Mar{\'\i}a and Valero-Mas, Jose J and Calvo-Zaragoza, Jorge},
  booktitle =   {{Proceedings of the 3rd International Workshop Pattern Recognition for Cultural Heritage}},
  pages     =   {93--107},
  year      =   {2022},
  publisher =   {Springer},
  address   =   {Montréal, Canada},
  month     =   aug,
  doi       =   {10.1007/978-3-031-37731-0_8},
}
```

## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033. Computational resources were provided by the Valencian Government and FEDER funding through IDIFEDER/2020/003.

## License
This work is under a [MIT](LICENSE) license.
