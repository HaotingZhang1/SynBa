# SynBa: Improved estimation of drug combination synergies with uncertainty quantification

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2023.01.24.524900-red)](https://www.biorxiv.org/content/10.1101/2023.01.24.524900)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyStan 2.19.0.0](https://img.shields.io/badge/PyStan-2.19.0.0-blueviolet)](https://pypi.org/project/pystan/2.19.0.0/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

This repository is the official implementation of **SynBa**, a method for the estimation of drug combination synergies with uncertainty quantification.
The [**preprint**](https://www.biorxiv.org/content/10.1101/2023.01.24.524900) that describes the method is now available on bioRxiv.

The implementation is built using PyStan, the Python interface to the probabilistic programming language **Stan**. This repository is being maintained and will be continuously updated.

### Interactive Notebooks
The repository contains `synba_mono.ipynb` ([**Colab link**](https://colab.research.google.com/github/HaotingZhang1/SynBa/blob/main/synba_mono.ipynb)) and `synba_combo.ipynb` ([**Colab link**](https://colab.research.google.com/github/HaotingZhang1/SynBa/blob/main/synba_combo.ipynb)), two interactive Colab/Jupyter notebooks that illustrate the monotherapy model (i.e. Box 1 in the paper) and the combination model (i.e. Box 2 in the paper) respectively.
These two notebooks can be run directly without setting up anything in advance.
Users can upload their own monotherapy/combination dose-response data, fit SynBa to their data and visualise the output including:
* (Both monotherapy and combination) The summary statistics table summarising the parameters in the fitted model
* (Monotherapy) The posterior distributions for the **potency (IC50)** and the **efficacy (Einf)**
* (Monotherapy) Sample dose-response curves from the posterior distribution
* (Monotherapy) The estimated noise level for the responses
* (Combination) The contour plot for the joint posterior distribution of the **synergistic efficacy (ΔHSA)** and the **synergistic potency (α)**
* (Combination) The histogram of the synergistic efficacy and the synergistic potency

### Reproducing results
The repository also contains the codes required to reproduce the results of our work (to be uploaded to bioRxiv). Results can be reproduced by following these steps:
* Download the repository.
* Set up a new environment and install the required packages. For example, it can be done by running the following:
```
conda create -n synba
conda activate synba
conda install pip
pip install -r requirements.txt 
```
* Figures 3 to 5 can then be reproduced by directly running `figure3.py`, `figure4.py` and `figure5.py`.
* To reproduce the quantitative results in Table 1 and Figures 6 and 7, the relevant datasets need to be downloaded. Both datasets used in this study are publicly available, although approval would be required to download the AstraZeneca-Sanger DREAM challenge, by submitting a data use statement. The DOI of the DREAM dataset is `10.7303/syn4231880`. The DOI of the subset of NCI-ALMANAC used in this study is `10.5281/zenodo.4135059`.
* To reproduce the results for NCI-ALMANAC, run `nci_extract_combo.py` to generate the numpy arrays for the dose-response matrices.
* After downloading the data (and running `nci_extract_combo.py` if NCI-ALMANAC is the dataset of interest), the quantitative results can then be reproduced by running `prediction_reproduce.py` and `calibration_reproduce.py`. Due to the large number of examples in the datasets, it will take a long time for the programme to complete.

### To be updated
The next updates of this repository will include the following:
* More visualisation options in the interactive notebooks.
* A more detailed documentation on how to fit SynBa to a dose-response dataset, as well as how to tune the priors and the other tunable settings.

### Citation
If you find SynBa useful in your research, please cite our paper:
```
@article{zhang2023synba,
	author = {Zhang, Haoting and Ek, Carl Henrik and Rattray, Magnus and Milo, Marta},
	title = {SynBa: Improved estimation of drug combination synergies with uncertainty quantification},
	elocation-id = {2023.01.24.524900},
	year = {2023},
	doi = {10.1101/2023.01.24.524900},
	URL = {https://www.biorxiv.org/content/early/2023/01/25/2023.01.24.524900},
	eprint = {https://www.biorxiv.org/content/early/2023/01/25/2023.01.24.524900.full.pdf},
	journal = {bioRxiv}
}
```
