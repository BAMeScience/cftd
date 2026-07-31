The Coarse-Fine Transport Distance (CFTD)
Scoring materials generative models with a distributional metric and optimal transport

This repository lets you easily score your materials generative model using the Coarse-Fine Transport Distance (CFTD). 
The CFTD is an extension of the Transport Novelty Distance and resolves the classical coverage-novelty tradeoff problem by employing two distinct featurizers. 
This decouples the judgement of physical and chemical quality of the generated materials from detecting memorized structures and allows a better evaluation of the generative models.  

The fine Identity Featurizer
This featurizer is a contrastive GNN trained with the InfoNCE loss, separating negative pairs of different structures from positive pairs created by augmentation. 
The contrastive learning objective learns a feature space that enables the CFTD to detect memorization, i.e., generated structures that are identical to the train set. 
We pre-trained one version on the MP20 training data, using rotations, translations, and Pettifor neighbors as augmentation operations. 
Additionally, small amounts of Gaussian noise were added to the lattice positions of the atoms, to detect generative models that just displace atoms by small amounts. 
The pre-trained version is provided in the checkpoints folder of this repository and can directly be used, 
if you trained your generative model on the MP20 train-val-test split from https://github.com/txie-93/cdvae/tree/main/data/mp_20.

The coarse MACE Featurizer 


A detailed discription is published on arxiv.
