# Characterizing Uncertainty in Machine Learning for Chemistry Scripts

Scripts associated with the paper "Characterizing Uncertainty in Machine Learning for Chemistry" (https://doi.org/10.26434/chemrxiv-2023-00vcg). They show how the data for this paper were calculated and how the figures are made.


Portions of the scripts and calculations are dependent on three other repositories.

* Chemprop. A software for using machine learning for chemical property prediction, using a Message Passing Neural Network model structure based on a 2-D graph representation of the molecule. The chemprop conda environment is necessary for most calculations carried out using scripts in this repository.
https://github.com/chemprop/chemprop

* Ensemble Projection. An software for using Bayesian inference to calculate what portion of a model's error is due to variance error and projecting an expected error level for other ensemble sizes.
https://github.com/cjmcgill/ensemble_projection

* Schnet. A software for using machine learning for chemical property prediction, operating on 3D molecular geometries.
https://github.com/atomistic-machine-learning/schnetpack


Two datasets are used in training models for the associated paper.

* Artificial Dataset of Molecular Enthalpies of Formation. A noise-free artificial dataset constructed for the study of model uncertainty (https://doi.org/10.5281/zenodo.7626488). The zenodo repository contains scripts for recreating the splits used in this study and adding noise to the dataset in the prescribed way.

* QM9. A DFT dataset comprising 134k small molecules constructed using the atoms C, H, O, N, F. (https://moleculenet.org/datasets-1). The QM9 targets, formatted as are used by the scripts for the paper, are provided in the data directory of this repository.


This repository contains the following scripts and directories that are necessary for making calculations and generating figures associated with this paper. For the method behind a particular figure, you can look at the associated script in the three scripts directories.

* calculation_scripts. A directory containing scripts for training the models and making predictions that are part of this paper. Scripts are highly parallelizable and this is recommended because of the large amount of computing time necessary to carry out all of the calculations.

* processing_scripts. A directory containing scripts that perform intermediate processing of the results.

* figure_scripts. A directory containing scripts that perform final processing of results and generate the figures used in the paper.

* data.zip. A compressed directory containing specific data splits used in some model training.

* custom_descriptors.zip. A compressed directory containing custom atomic descriptors used for training the models associated with Figure 6 that used altered descriptors in message passing.

* qm9.zip. A compressed file that, after being extracted, is needed for training schnet on the qm9 dataset.


Here are brief descriptions of the eleven figures presented in the paper.

* Figure 1. A comparison of how model performance changes when noise is present in the training and test sets. Uses the artificial noise-free dataset.

* Figure 2. A demonstration of the similar aleatoric limit behavior present in model performance trained and evaluated on noisy data when compared between different noise distributions. Uses the artificial noise-free dataset.

* Figure 3. A demonstration of mean-variance estimation being used to distinguish different magnitudes of noise applied systematically to different data regimes. Uses the artificial noise-free dataset.

* Figure 4. Model performance trends with dataset size and model hidden size, broken down into components due to variance error and not due to variance error. Uses the artificial noise-free dataset.

* Figure 5. Demonstration of differential model performance when used on targets that are dependent on 3D geometries or are intrinsic/extrinsic properties of molecule size. Uses QM9 dataset. Compares Chemprop and Schnet implementations of models.

* Figure 6. Demonstration of differing performance based on the model input features. Compares morgan fingerprint performance to d-MPNN models, including instances where the d-MPNN models are using intentionally poor featurization of atoms. Uses QM9 dataset.

* Figure 7. Demonstration of model improvements due to ensembling under different model and data conditions. Uses the artificial noise-free dataset.

* Figure 8. Quantification of uncertainty predictions made using ensemble variance according to the AUCE metric, as compared for different model conditions. Uses the artificial noise-free dataset.

* Figure 9. Comparison of the scores reported from cross-validation models versus ensemble models. Uses the artificial noise-free dataset.

* Figure 10. Demonstration of model performance when extrapolating to different molecule sizes and how the performance differs for mean versus sum aggregation. Uses the QM9 dataset.

* Figure 11. Demonstration of model performance with data leakage between test and training sets.
