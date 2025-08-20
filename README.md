# Advancing Graph Isomorphism Tests with Metric Space Indicators: A Tool for Improving Graph Learning Tasks
This repository contains the code for the CIKM 2025 submission:

### Overview:
The MSIGNN introduces an effective metric indicator based GNN. The key contributions of our MSIGNN are twofold: (1) a novel isomorphism test k-MSI; and (2) a GNN based on node metric indicator, leading to enhanced graph classification performance in GNNs.

### About
* The folder BREC_GNN is the code for the graph isomorphism test, you should change the dataset in BRECDataset_v3.py. All datasets are progressed by node metric indicator.
* The folder BREC_NONGNN is the code for the MSI to detect the capability of the graph isomorphism.
* The folder TUDataset is to test the performance of graph classification by MSIGNN.

### Requirements
Code is written in Python 3.8 and requires:
* PyTorch   1.9.0
* NetworkX  2.3
* torch  1.13.1+cu117
* torch-geometric   2.3.1


## Cite as
> Shenghui Zhang, Pak Lon Ip, Rongqin	Chen, Shunran Zhang, and Leong Hou U, Advancing Graph Isomorphism Tests with Metric Space Indicators: A Tool for Improving Graph Learning Tasks. CIKM submission 2025.

### Bibtex:
```
@inproceedings{zhang2025CIKM,
  title={Advancing Graph Isomorphism Tests with Metric Space Indicators: A Tool for Improving Graph Learning Tasks},
  author={Shenghui Zhang, Pak Lon Ip, Rongqin	Chen, Shunran Zhang, and Leong Hou U},
  booktitle={CIKM},
  year={2025}
}

