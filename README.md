# Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs

Jérémie Dentan<sup>1</sup>, Davide Buscaldi<sup>1, 2</sup>, Sonia Vanier<sup>1</sup>

<sup>1</sup>LIX (École Polytechnique, IP Paris, CNRS) <sup>2</sup>LIPN (Sorbonne Paris Nord)

## Presentation of the repository

This repository implements the experiments of our preprint "Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs".

### Abstract of the paper

Verbatim memorization in Large Language Models (LLMs) is a multifaceted phenomenon involving distinct underlying mechanisms. We introduce a novel method to analyze the different forms of memorization described by the existing taxonomy. Specifically, we train Convolutional Neural Networks (CNNs) on the attention weights of the LLM and evaluate the alignment between this taxonomy and the attention weights involved in decoding.

We find that the existing taxonomy performs poorly and fails to reflect distinct mechanisms within the attention blocks. We propose a new taxonomy that maximizes alignment with the attention weights, consisting of three categories: memorized samples that are *guessed* using language modeling abilities, memorized samples that are *recalled* due to high duplication in the training set, and *non-memorized* samples. Our results reveal that few-shot verbatim memorization does not correspond to a distinct attention mechanism. We also show that a significant proportion of extractable samples are in fact guessed by the model and should therefore be studied separately. Finally, we develop a custom visual interpretability technique to localize the regions of the attention weights involved in each form of memorization.

### License and Copyright

Copyright 2025-present Laboratoire d'Informatique de Polytechnique. Apache Licence v2.0.

Please cite this work as follows:

```bibtex
@misc{dentan_guess_2025,
	title = {Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs},
	url = {https://arxiv.org/abs/2508.02573},
	author = {Dentan, Jérémie and Buscaldi, Davide and Vanier, Sonia},
	month = aug,
	year = {2025},
}
```

## Reproducing Experiments

### Important: 45 TB of storage required

Storing the attention weights demands extremely large disk space. Running our full set of experiments requires **45 TB** of storage: approximately **30 TB** for the taxonomy benchmark and **15 TB** for the optimization of `rho`. We strongly recommend running the experiments on an HPC cluster with sufficient disk capacity.

We provide a script in `scripts/data_management` to sync the key outputs of your computation to a local machine. For our full experiments, this script requires around **10 GB** of local storage.

### Step 1: Set up the Python environment

Our experiments were conducted with **Python 3.11.0**. We provide a `requirements.txt` file listing the necessary dependencies. We also include `env_macos.yaml` and `env_redhat.yaml` files that specify the full configurations of our local and remote environments, respectively.

### Step 2: Compute attention weights and train CNNs

To compute the attention weights and train CNNs under the taxonomy setup, use the following SLURM scripts on your HPC cluster:

* `scripts/benchmark_taxonomies.sh`: Benchmarks the taxonomies (used for Tables 1, 3, 5, 6, and 7).
* `scripts/optimize_rho.sh`: Benchmarks different values of the `rho` hyperparameter (used for Table 4).

### Step 3: Sync data locally

**Sync selected outputs:** Use `scripts/data_management/hpc_sync.sh` to download around **10 GB** of output locally. This excludes the large `attention_patterns.h5py` files but includes the most relevant results for figures. Most figures can be reproduced from these files. Some experiments that rely directly on attention weights must still be run remotely.

**Weights & Biases (wandb):** We used `wandb` to track CNN training and log performance. Computations on the HPC are performed offline. Once finished, sync the logs using `scripts/data_management/hpc_sync.sh`, then push them to your `wandb` online account. To do so:

1. Run `wandb login` in your terminal.
2. From the `output` directory, run `wandb sync --sync-all`.

### Step 4: Reproduce the figures

* **Figure 1:** Hand-crafted. See `figures/00_diagrams.pptx`
* **Figure 2:** See `figures/01_explore_categories.ipynb`
* **Figure 3:** Left: extracted attention weights (`figures/02_explore_patterns.ipynb`), Right: hand-crafted (`figures/00_diagrams.pptx`)
* **Figure 4:** Hand-crafted. See `figures/00_diagrams.pptx`
* **Tables 1, 3, 5, 6, 7** and **Figures 5, 7-12:** See `figures/03_wandb_explore.ipynb`
* **Table 4:** See `figures/04_ROUGE_optimization.ipynb`
* **Figures 6, 13–18:** See `figures/05-interpretation_rasy.ipynb`

Some notebooks require direct access to the full attention weights and must be executed on the HPC cluster. Others can be run locally using the minimal data synced by `scripts/data_management/hpc_sync.sh`. The appropriate setting is indicated in the first cells of each notebook.

We also provide `figures/06_extract_patterns.ipynb`, which can be run remotely to extract a limited number of attention weights for local exploration.

## Acknowledgements

This work received financial support from Crédit Agricole SA through the research chair "Trustworthy and responsible AI" with École Polytechnique.

This work was granted access to the HPC resources of IDRIS under the allocation 2023-AD011014843R1 made by GENCI.
