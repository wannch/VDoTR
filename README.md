# VDoTR
This respository is the source code for the accepted paper [VDoTR: Vulnerability Detection Based on Tensor Representation of Comprehensive Code Graphs](https://www.sciencedirect.com/science/article/pii/S0167404823001578).

## Abstract of Our Work
Code vulnerability detection has long been a critical issue due to its potential threat to computer systems. It is imperative to detect source code vulnerabilities in software and remediate them to avoid cyber attacks. To automate detection and reduce labor costs, many deep learning-based methods have been proposed. However, these approaches have been found to be either ineffective in detecting multiple classes of vulnerabilities or limited by treating original source code as a natural language sequence without exploiting the structural information of code. In this paper, we propose VDoTR, a model that leverages a new tensor representation of comprehensive code graphs, including AST, CFG, DFG, and NCS, to detect multiple types of vulnerabilities. Firstly, a tensor structure is introduced to represent the structured information of code, which deeply captures code features. Secondly, a new Circle Gated Graph Neural Network (CircleGGNN) is designed based on tensor for hidden state embedding of nodes. CircleGGNN can perform heterogeneous graph information fusion more directly and effectively. Lastly, a 1-D convolution-based output layer is applied to hidden embedding features for classification. The experimental results demonstrate that the detection performance of VDoTR is superior to other approaches with higher accuracy, precision, recall, and F1-measure on multiple datasets for vulnerability detection. Moreover, we illustrate which code graph contributes the most to the performance of VDoTR and which code graph is more sensitive to represent vulnerability features for different types of vulnerabilities through ablation experiments.

## Citation
@article{FAN2023103247,
title = {VDoTR: Vulnerability detection based on tensor representation of comprehensive code graphs},
journal = {Computers & Security},
volume = {130},
pages = {103247},
year = {2023},
issn = {0167-4048},
doi = {https://doi.org/10.1016/j.cose.2023.103247},
url = {https://www.sciencedirect.com/science/article/pii/S0167404823001578},
author = {Yuanhai Fan and Chuanhao Wan and Cai Fu and Lansheng Han and Hao Xu},
keywords = {Source code vulnerability detection, Tensor-based feature, GGNN, Code graphs, Heterogeneous information fusion}
}