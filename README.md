
<h1 align="center">Few‑shot learning for COVID‑19 chest X‑ray classification with imbalanced data: an inter vs. intra domain study</h1>

<h4 align="center">Full text available <a href="https://doi.org/10.1007/s10044-024-01285-w" target="_blank">here</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/Tensorflow-%FFFFFF.svg?style=flat&logo=Tensorflow&logoColor=orange&color=white" alt="Tensorflow">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#license">License</a>
</p>


## About

Inter and Intra-domain study in few-shot learning scenarios with severe data imbalance based on Siamese neural networks. Tested over four chest X-ray datasets with annotated cases of both positive and negative COVID-19 diagnoses. All datasets are publicly accessible: ChestX-ray can be found at [`https://nihcc.app.box.com/v/ChestXray-NIHCC`](https://nihcc.app.box.com/v/ChestXray-NIHCC), GitHub-COVID at [`https://github.com/ieee8023/covid-chestxray-dataset`](https://github.com/ieee8023/covid-chestxray-dataset), PadChest is available at [`https://bimcv.cipf.es/bimcv-projects/padchest`](https://bimcv.cipf.es/bimcv-projects/padchest), and BIMCV-COVID repositories can be accessed through [`https://bimcv.cipf.es/bimcv-projects/bimcv-covid19`](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19).


## How To Use

To replicate the work, execute the file [`main_launch_experiments.py`](main_launch_experiments.py). It is ready to receive different parameters, each one corresponding to a concrete experiment.

The code has been used over a Docker environment. However, the requirements for any other virtual environment can be easily extracted from [`docker/Dockerfile`](docker/Dockerfile).


## Citations

```bibtex
﻿@Article{Galan-Cuenca2024,
  author={Galan-Cuenca, Alejandro
  and Gallego, Antonio Javier
  and Saval-Calvo, Marcelo
  and Pertusa, Antonio},
  title={Few-shot learning for COVID-19 chest X-ray classification with imbalanced data: an inter vs. intra domain study},
  journal={Pattern Analysis and Applications},
  year={2024},
  month={Jun},
  day={11},
  volume={27},
  number={3},
  pages={69},
  issn={1433-755X},
  doi={10.1007/s10044-024-01285-w},
  url={https://doi.org/10.1007/s10044-024-01285-w}
}


```

## License
This work is under a [MIT](LICENSE) license.
