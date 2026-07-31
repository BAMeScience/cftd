# The Coarse-Fine Transport Distance (CFTD)
**Scoring materials generative models with a distributional metric and optimal transport.**

The Coarse-Fine Transport Distance (CFTD) lets you easily score your materials generative model. 
It is an extension of the Transport Novelty Distance and resolves the coverage-novelty tradeoff problem by employing two distinct featurizers. 
This decouples the judgement of physical and chemical quality of the generated materials from detecting memorized structures and allows a better evaluation of the generative models.  

## How it works

### The fine Identity Featurizer
This featurizer is a contrastive GNN trained with the InfoNCE loss, separating negative pairs of different structures from positive pairs created by augmentation. 
The contrastive learning objective learns a feature space that enables the CFTD to detect memorization, i.e., generated structures that are identical to the train set. 
We pre-trained one version on the MP20 training data, using rotations, translations, and Pettifor neighbors as augmentation operations. 
Additionally, small amounts of Gaussian noise were added to the lattice positions of the atoms, to detect generative models that just displace atoms by small amounts. 
The pre-trained version is provided in the checkpoints folder of this repository and can directly be used, 
if you trained your generative model on the MP20 train-val-test split from https://github.com/txie-93/cdvae/tree/main/data/mp_20.

### The coarse MACE Featurizer 
This featurizer uses the invariant, hidden layers from a pre-trained MACE foundation model (MACE-MP-0b3) to evaluate the quality of the generated materials. 
The layer are mean pooled and randomly projected to create the final feature space. This compressed representation of MACE MLIPs is a good indicator of the general physical and chemical quality of a structure, although its coarse nature is not a perfect representation of the materials parameters.

Both featurizers are combined and used in an Optimal Transport framework to penalize memorized and "low quality" materials simultaneously within a single metric. 
Generally, lower CFTD values indicate better perfomance.
A detailed discription is published on arxiv.


## Evaluating your model with the CFTD

### Installation
We recommend using uv:

```bash
# Navigate to the respective project folder, then clone this repository
git clone https://github.com/BAMeScience/cftd.git

# Create python environment
cd cftd
uv sync
uv pip list
```

Without uv:
```bash
# Navigate to the respective project folder, then clone this repository
git clone https://github.com/BAMeScience/cftd.git

# Create python environment
cd cftd
python -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install --upgrade pip
pip install .
```

### Scoring your model with CFTD

#### Model trained on MP20
If you trained your model on the MP20 train-val-test split from https://github.com/txie-93/cdvae/tree/main/data/mp_20, you can directly load the pre-trained GNN and initialize the CoarseFineTransportDistance.
  
  ```bash
# load the pretrained model
id_feat_gnn = EquivariantCrystalGCN(hidden_dim=128, num_rbf=128, n_layers=3).to(device)
id_feat_gnn.load_state_dict(encoder_state)
id_feat_gnn.eval()

# define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize the CoarseFineTransportDistance
cftd = CoarseFineTransportDistance(
      train_structs = train_mp20, # path to the .csv that includes the MP20 train data
      calib_structs = val_mp20, # path to the .csv that includes the MP20 validation data
      geo_model = id_feat_gnn, # load the GNN of the identity featurizer
      mace_model_name = "medium-0b3", # indicate the MACE version to use for the MACE featurizer
      coupling_alpha = 0.5, # Weighting of Identity and MACE featurizer
      beta_goldilocks = 0.6, # Proportion of samples penalized in novelty and quality regimes
      batch_size = 32, 
      device=device,
  )

  # call cftd.computeCFTD to calculate the total CFTD value, as well as the individual quality and memorization components
  total, qual, mem = cftd.computeCFTD(structs) # structs are a sequence of your generated structures
  ```

#### Model trained on a different dataset
In case you used a different dataset, it is recommended to 
- Retrain the contrastive GNN by executing train_mp20.py in scripts with the train-val-test split of your data 
- Change the respective lines in the above code to load your new model
- Proceed as shown above

If you want to retrain the contrastive GNN either way, you can download the MP20 data by calling download_mp20.py

### Compare different models
To compare your model to other published generative models, call download_xtalmet_models.py and calculate the CFTD for all models relevant for you. All of these models were trained on MP20, you can therefore adapt the code above by changing the "structs" to the respective model results when calling cftd.computeCFTD(structs). 
Lower CFTD values generally indicate better models, in the sense that the quality distribtion of the training set is covered well, while the generated structures are not memorized directly from the train set.

