---
layout: notes
section-type: notes
title: Machine Learning with Graphs
category: ml
---

* TOC
{:toc}
---
* History Papers
    * A new model earning in graph domains

* Course Material
    * Course [CS 224w: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
    * Course [Video Link](https://www.youtube.com/watch?v=0eNQnc0eOB4&list=PL1OaWjIc3zJ4xhom40qFY5jkZfyO5EDOZ)


## 1. Introduction and Structure of Graphs
<hr>

### 1.1 Components of Graph
* Objects: nodes, vertices $N$
* Interactions: links, edges $E$
* System: network, graph $G(N,E)$

### 1.2 Features of Graph
* Avg Degree $\bar{k}=\frac{1}{N}{\sum_{i=1}^{N}k_i}=\frac{2E}{N}$
* Adjacency Matrix: Undirected and Directed Graphs

### 1.3 Edge Attributes
* Weight(e.g. frequency of communication)
* Ranking(best friend, second best friend)
* Type(friend, relative, co-worker)
* Sign(Friend vs. Foe, Trust vs. Distrust)

### 1.4 Types of Graphs
* Unweighted; Weighted
* Self-edges; Multigraph


<br>
<br>
<br>

## 2. Properties of Networks and Random Graph Models
<hr>

### 2.1 Notations
* Degree Distribution: $P(k)$
* Path Length: $h$
* Clustering Coefficient: $C$
* Connected Components: $s$

### 2.2 Paths in a Graph
* Distance in a Graph
* Network Diameter: The maximum(shortest path) distance between any pair of nodes in a graph
* Average path length;

### 2.3 Clustering Coefficient $C_i$
* Undirected Graph Clustering Coefficient
    * How connected are $i$'s neighbours to each other?
    * Node $i$ with degree $k_i$
    * $C_i = \frac{2e_i}{k_i(k_i-1)}$, denominator show the maximum edge connections for neighbour nodes of node $i$

<center>
<img class = "large" src=".//graph/001.png" height="65%" width="65%">
</center>

### 2.4 Connectivity
* Size of largest connected component
* Largest component = Giant Component

### 2.5 Propertities of $G_{np}$
* Degree distribution: $p(k)=C_{n-1}^{k}p^k(1-p)^{n-1-k}$
* Clustering Coefficient of $G_{np}$: $C=p=\bar{k}/n$
* Averge Path Length: $O(\log{n})$


### 2.5 Small-World Model
* Can we have high clustering while also having short paths?

<center>
<img class = "large" src=".//graph/002.png" height="65%" width="65%">
</center>


<br>
<br>
<br>

## 3. Motifs and Structure Roles in Networks
<hr>

### 3.1 Subnetwork
### 3.2 Network Motifs
* Motif: Recurring, Significant, Patterns of interconnections
* And motif occur in the real network more often than the random network.
* $Z_i$ captures the siginificance of motif i
* Network Significance Profile(SP):
    
### 3.3 Graphlets
* Graph Degree Vector
* Automorphism Orbits

### 3.4 Graph Isomophism
* Example: Are $G$ and $H$ topologically equivalent?

<br>
<br>
<br>

## 4. Graph Representation Learning
<hr>

Following fields are **Graph Representation Learning** focused on:
* Node Classification
* Link Prediction

### 4.1 Feature Learning in Graphs
* Feature Representation Embedding
* Task: Map each node in a network into a low-dimensional space

### 4.2 Node Embedding
* CNN for fixed-size images/grids
* RNNs or word2vec for text/sequences

### 4.3 Embedding Nodes Task
<center>
<img class = "large" src=".//graph/004.png" height="65%" width="65%">
</center>

Hence, we can analyse the similarity of those nodes in space, and we have many approaches to measure the distance like Eucliden Distance, Cos Vector etc. In this way, we can just use the 

### 4.4 Random Walk Approaches to Node Embeddings
* $z_u^{T}z_v$ probability that $v$ and $u$ co-occur on a random walk over the network

### 4.5 Unsupervised Feature Learning
<center>
<img class = "large" src=".//graph/005.png" height="65%" width="65%">
</center>

<center>
<img class = "large" src=".//graph/006.png" height="65%" width="65%">
</center>

<br>
<br>
<br>

## 5. Graph Neural Network
<hr>

### [Lecture video](https://www.youtube.com/watch?v=7JELX6DiUxQ)
### 5.1 Nodes Embeddings
<center>
<img class = "large" src=".//graph/007.png" height="65%" width="65%">
</center>

* Two Key Components:
    * Encoder
    * Similarity Function

### 5.2 Basics of Deep Learning for Graphs
* Idea: Neighbourhood Aggregation
<center>
<img class = "large" src=".//graph/008.png" height="65%" width="65%">
</center>

* Final layer $h^{K}_{v}$ is embedding of $\mathbf{z}_{v}$
* Train the Model
    * $\mathbf{W}_{k}$
    * $\mathbf{B}_{k}$
* Inductive Capability
* So far, the GraphNN aggregate the neighbour messages by taking their(weighted) average 

### 5.3 GraphSAGE Graph Neural Network Architecture
* Concatenece:
    * Concatenate neighbour embedding and self embedding
    * Unlike Graph Convolution with adding itself, we just concatenate itself features then activate with non-linearity function
* Aggregation:
    * Use generalized aggregation function
    * Unlike Graph Convolution with just average

<center>
<img class = "large" src=".//graph/009.png" height="65%" width="65%">
</center>

* Aggregation Variants
    * Generally, there are several ways to implement aggregate
    * Mean
    * Pool (mean or max across a coordinate)
    * LSTM (make model much deeper with LSTM)
<center>
<img class = "large" src=".//graph/010.png" height="65%" width="65%">
</center>
Hints: we can apply different pooling startegies

### 5.4 Implementation
<center>
<img class = "large" src=".//graph/011.png" height="65%" width="65%">
</center>

* Notation:
    * $D$ is degree matrix
    * $A$ is adjeacent matrix
    * $H^{k-1}$ is message matrix from previous layer

* $D^{-1}$ matrix acts as a mean function in this formula.
* $AH^{k-1}$ is aimed to sum all neighbour features

### 5.5 Graph Attention Network (GAT)
* Simple Neighbourhood Aggregation in Graph Convolution
    * Use coefficient of ${\alpha}_{vu}$
    * All neighbour $u \in {N}(v)$ are equally important to node $v$
<center>
<img class = "large" src=".//graph/012.png" height="65%" width="65%">
</center>

* Attention Mechanism
    * Use $e_{vu}$ as coefficient
    * Mechanism $a$ may achieve in different ways including Simple Single-Layer Neural Network
<center>
<img class = "large" src=".//graph/013.png" height="65%" width="65%">
</center>

<center>
<img class = "large" src=".//graph/014.png" height="65%" width="65%">
</center>

### 5.6 Papers on GNN
* Tutoiral and Overviews

    * [Relational inductive biases and graph networks (Battaglia et al., 2018)](https://arxiv.org/pdf/1806.01261.pdf)
    * [Representation learning on graphs: Methods and applications (Hamilton et al., 2017)](https://arxiv.org/pdf/1709.05584.pdf)

* Attention-based neighborhood aggregation
    * [VAIN: Attentional Multi-agent Predictive Modeling (Hoshen, 2017)](https://papers.nips.cc/paper/6863-vain-attentional-multi-agent-predictive-modeling.pdf)
    * [Graph Attention Networks (Velickovic et al., 2018)](https://arxiv.org/pdf/1710.10903.pdf)
    * [Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation (Liu et al., 2018)](https://arxiv.org/pdf/1809.09078.pdf)

* Embedding entire graphs
    * [Hierarchical Graph Representation Learning withDifferentiable Pooling(Ying et al., 2018)](http://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf)
    * [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models(You et al., 2018)](https://arxiv.org/pdf/1802.08773.pdf)
    * [Neural Relational Inference for Interacting Systems(Kipfet al., 2018)](https://arxiv.org/pdf/1802.04687.pdf)
    * [How powerful are graph neural networks (Xu et al., 2017)](https://arxiv.org/pdf/1810.00826.pdf)

* Embedding nodes:
    * [Representation Learning on Graphs with Jumping Knowledge Networks (Xu et al., 2018)](https://arxiv.org/pdf/1806.03536.pdf)
    * [Position-aware GNN (You et al. 2019)](https://arxiv.org/pdf/1906.04817.pdf)

* Spectral approaches to graph neural networks:
    * [Deep Convolutional Networks on Graph-Structured Data (Bruna et al. 2015)](https://arxiv.org/pdf/1506.05163.pdf)
    * [Convolutional Neural Networks on Graphswith Fast Localized Spectral Filtering (Defferrard et al., 2016)](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf)
    * [Geometric deep learning:going beyond Euclidean data (Bronstein et al., 2017)](https://arxiv.org/pdf/1611.08097.pdf)
    * [Geometric deep learning on graphs and manifolds using mixture model CNNs (Monti et al., 2017)](http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf)
    
* Other GNN Techniques
    * [Pre-Training Graph Neural Networks for Generic Structural Feature Extraction (Hu et al., 2019)](https://arxiv.org/pdf/1905.12265.pdf)
    * [GNNExplainer: Generating Explanationsfor Graph Neural Networks (Ying et al., 2019)](http://papers.nips.cc/paper/9123-gnnexplainer-generating-explanations-for-graph-neural-networks.pdf)
