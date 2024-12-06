# MotifGT-DTI: pivotal motif-based graph transformer model improves drug-target prediction via bilinear attention network


## Introduction
**MotifGT-DTI** is a novel motif-based model with graph transformer for drug-target interaction prediction. 
Specifically, **MotifGT-DTI** captures complex molecular patterns of molecular motif subgraphs and protein 3D pocket subgraphs with graph transformer.
To attain protein characteristics more comprehensively, **MotifGT-DTI** fuses 1D sequence and 3D structure features with cross-attention from two views 
and predicts potential interactions with bilinear attention.
Then, the structural-level association patterns of drug molecules and proteins are connected via a bilinear attention network.

## Framework
<!---[MotifGT-DTI](imgs/MotifGT-DTI.png)--->!
## System Requirements
The source code developed in Python 3.11 using PyTorch 2.0.1. 
The required python dependencies are given in requirements.txt. 

## Installation Guide
Clone this Github repo and set up a new conda environment. 
```
# create a new conda environment
$ conda create --name motifgt python=3.11
$ conda activate motifgt

# install requried python dependencies
$ pip install -r requirements.txt

# clone the source code of MotifGT-DTI
$ git clone https://github.com/Dimpleney/MotifGT-DTI.git
```


## Data
### Datasets
The `data` folder contains all experimental data used in MotifGT-DTI, including Human,BioSNAP and Drugbank.

### Data_Preprocess
Since this experiment needs to use to the .pdb file, the first run may take a long time. 
It is recommended to run feature_pre.py for feature preprocessing after downloading all pdb files to speed up the subsequent training.


## Run DrugBAN on Our Data to Reproduce Results
To train MotifGT-DTI, you just need to run main.py.



## Acknowledgements
This implementation is inspired and partially based on earlier works [1], [2] and [3].


## References
    [1] Bai P, Miljković F, John B, et al. Interpretable bilinear attention network with domain adaptation improves drug–target prediction[J]. Nature Machine Intelligence, 2023, 5(2): 126-136.
    [2] Khodabandeh Yalabadi A, Yazdani-Jahromi M, Yousefi N, et al. FragXsiteDTI: Revealing Responsible Segments in Drug-Target Interaction with Transformer-Driven Interpretation[C]//International Conference on Research in Computational Molecular Biology. Cham: Springer Nature Switzerland, 2024: 68-85.
    [3] Wu H, Liu J, Jiang T, et al. AttentionMGT-DTA: A multi-modal drug-target affinity prediction using graph transformer and attention mechanism[J]. Neural Networks, 2024, 169: 623-636.
