# SynBa: Improved estimation of drug combination synergies with uncertainty quantification

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

This repository is the implementation of SynBa in Python 3.7. It will be continuously updated in the next few weeks.

Currently, the repository contains the codes required to reproduce the results of our work (to be uploaded to bioRxiv). Results can be reproduced by following these steps:
* Download the repository.
* Set up a new environment and install the required packages. For example, it can be done by running the following:
```
conda create -n synba
conda activate mfcvae
conda install pip
pip install -r requirements.txt 
```
* Figures 3 to 5 can then be reproduced by directly running `figure3.py`, `figure4.py` and `figure5.py`.
* To reproduce the quantitative results in Table 1 and Figures 6 and 7, the relevant datasets need to be downloaded. Both datasets used in this study are publicly available, although approval would be required to download the AstraZeneca-Sanger DREAM challenge, by submitting a data use statement. The DOI of the DREAM dataset is `10.7303/syn4231880`. The DOI of the subset of NCI-ALMANAC used in this study is `10.5281/zenodo.4135059`.
* After downloading the data, the quantitative results can then be reproduced by running `prediction_reproduce.py` and `calibration_reproduce.py`. Due to the large number of examples in the datasets, it will take a long time for the programme to complete.

The next updates of this repository will include the following:
* An interactive Jupyter/Colab Notebook where users can upload their own dose-response data, fit SynBa to their data and visualise the output.
* A step-by-step instruction on how to fit SynBa to a dose-response dataset.
