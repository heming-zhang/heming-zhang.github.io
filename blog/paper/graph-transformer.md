---
layout: notes
section-type: notes
title: Transformers in Graph Neural Network
category: graph transformer
---

* TOC
{:toc}
---

# 0. A Generlization of Transformer Networks to Graphs
<hr>

* [Paper link](https://arxiv.org/pdf/2012.09699v2.pdf)

Following are the comprehensive reviews for the papers appeared in the application of the transformer networks to graphs.


## 1. Graph Transformer
<hr>

* [Paper Link](https://openreview.net/pdf?id=HJei-2RcK7)
ICLR 2019

<center>
<img class="center medium" src=".//paper-pictures-graph-transformer/figure01.png" width="100%">
</center>

### 1.1 Improvements compared with GAT

(1) GTR attends to all graph nodes at every graph update disregard of whether two nodes are directly connected or not, while GAT only attends to directed connected neighboring nodes for a considered node by a proposed masked attention. 

(2) GTR, on the other hand, is able to capture global context as which is significant for modeling long-range relations and fasten graph learning by allowing information propagation between implicitly connected nodes.

(3) GTR incorporates both prior graph structure and graph structure learning where using prior edge weights allows the efficient utilization of prior knowledge.

(4) GTR is able to transform representations across various graph structures while GAT is restricted to the same graph propagation.v 


## 2. Universal Graph Transformer Self-Attention Networks
<hr>

* [Paper Link](https://arxiv.org/pdf/1909.11855.pdf)
WWW 2022

### 2.1 Introduction
* We propose a transformer-based GNN model, named UGformer, to learn graph representations. In particular, we consider two model variants of (1) leveraging the transformer on a set of sampled neighbors for each input node and (2) leveraging the transformer on all input nodes.

* The unsupervised learning is essential in both industry and academic applications, where expanding unsupervised GNN models is more suitable to address the limited availability of class labels. Thus, we present an unsupervised transductive learning  approach to train GNNs.

* Experimental results show that the first UGformer variant obtains state-of-art accuracies on social network and bioinformatics datasets for graph classification in both inductive setting and unsupervised transductive setting; 

<center>
<img class="center medium" src=".//paper-pictures-graph-transformer/figure02.png" width="100%">
</center>

### 2.2 Models
* For Variant1, they leveraged the transformer on a set of sampled neighbors for each node

* For Variant2, they leveraged the transformer on all input nodes

Therefore, the Variant1 is suitable for the large scale graph. The Variant2 aimed to leverage the transformer for small and medium graphs.

## 3. GRAPH-BERT: Only Attention is Needed for Learning Graph Representations
<hr>

* [Paper Link](https://arxiv.org/pdf/2001.05140.pdf)
Published Jan 22, 2020

### 3.1 Summary of the contributions
* **New GNN Model**: In this paper, we introduce a new GNN model GRAPH-BERT for graph data representation learning. GRAPH-BERT doesn't rely on the graph links for representation learning and can effectively address the suspended animation probelms aforementioned. Also GRAPH-BERT is trainable with sampled linkless subgraphs (i.e., target node withg context), which is more efficient than exisiting GNNs constructed for the complete input graph. To be more precise, the training cost of GRAPH-BERT is only decided by (1) training instance number, and (2) sampled subgraph size, which is uncorrelated with the input graph size at all.

* **Unsupervised Pre-Training**: Given the input unlabeled graph, we will pre-train GRAPH-BERT based on to two common tasks in graph studies, i.e., node attribute reconstruction and graph structure recovery. Node attribute recovery ensures the learned node representations can capture the input attribute information; whereas graph structure recovery can further ensure GRAPH-BERT learned with linkless subgraphs can still maintain both the graph local and global structure properties.

* **Fine-Tuning and Transfer**: Depending on the specific application task objectives, the GRAPH-BERT model can be further fine-tuned to adapt the learned representaions to specific application requirements, e.g., node classification and graph clustering. Meanwhile, the pre-trained GRAPH-BERT can also be transferred and applied to other sequential models, which allows the construction of functional pipelines for graph learning.

### 3.2 Background introduction
* **BERT and TRANSFORMER**: In NLP, the dominant sequence transduction models are based on complext recurrent or convolutional neural networks. However, the inherently sequyential nature precludes parallelization within training examples. Therefore, in [Vaswani et al., paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), the authors propose a new network architecture, the TRANSFORMER, based solely on attention mechanisms, dispensing with recurreence and convolutions entirely. With TRANSFORMER, [Devlin et al., 2018](https://arxiv.org/pdf/1810.04805.pdf) further introduces BERT for deep language understanding, which obtains new state-of-art results in eleven natural language processing tasks. In recent years, TRANSFORMER and BERT based learning approaches have been used extensively in variously learning tasks.

### 3.3 Model
<center>
<img class="center medium" src=".//paper-pictures-graph-transformer/figure04.png" width="100%">
</center>

## 4. Graph Transformer Networks
<hr>

* [Paper link](https://proceedings.neurips.cc/paper_files/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf)
NIPS 2019

### 4.1 Introduction
* **Current Limitations**: One limitation of most GNNs is that they assume the graph structure to operate GNNs on is fixed and homogeneous. Since the graph convolutions discussed above are determined by the fixed graph structure, a noisy graph with missing/spurious connections results in ineffective convolution with wrong neighbors on the graph.

* A naive approach is to ignore the node/edge types and treat them as in a homogeneous graph (a standard graph with one type of nodes and edges). This, apprently, is suboptimal since models canot exploit the type information. A more recent remedy is to mannually design meta-paths, which are paths connected with heterogenenous edges, and transform a heteroegeneous graph into a homogeneous graph defined by the meta-paths. The convolutional GNNs can operate on the transformed homogeneous graphs. This is a two-stage approach and requires hand-crafted meta-paths for each problem. **The accuracy of downstream analysis can be significantly affected by the choice of these meta-paths.**

* Here, we develop Graph Transformer Network (GTN) that learns to transform a heterogeneous input graph into useful meta-path graphs for each task and learn node representation on the graphs in an end-to-end fashion. GTNs can be viewed as a graph analogue of Spatial Transformer Networks, which explicity learn spaptial transformations of input images or features. **The main challenge to transform a heterogeneous graph into new graph structure defined by meta-paths is that meta-paths may have an arbitrary length and edge types.** For example, author classification in citation network may benefit from meta-paths which are Author-Paper-Author (APA) or Author-Paper-Conference-Paper-Author (APCPA). Also, the citation networks are directed graphs where relatively less graoh neural networks can operate on.

### 4.2 Contributions of GTN
Our contributions are as follows: (1) We propose a novel framework Graph Transformer Networks, to learn a new graph structure which involves identifying useful meta-paths and multi-hop connections for learning effective node representation on graphs. (2) The graph generation is interpretable and the model is able to provide insight on effective meta-paths for predicition. (3) We prove the effectiveness of node representation learnt by Graph Transformer Networks resulting in the best performance against state-of-art methods that additionally use domain knowledge in all three benchmark node classification on heterogeneous graph.

### 4.3 Method
* **Meta-Path** denoted by $p$ is path on the heterogenous graph $G$ that is connected with heterogeneous edges, i.e., ${v_1}\rightarrow{v_2}\rightarrow{\cdots}\rightarrow{v_{l+1}}$, where $t_{l}\in{\mathcal{T}^{e}}$ denotes an $l$-th edge type of meta-path. It defines a composite relation $R=t_1\circ{t_2}\cdots\circ{t_l}$ between node $v_1$ and $v_{l+1}$, where $R_1{\circ}{R_2}$ denotes the composition of relation $R_1$ and $R_2$. Given the composite relation $R$ or the sequence of edge types $(t_1, t_2, \cdots, t_l)$, the adjacency matrix $A_p$ of the meta-path $P$ is obtained by the multiplications of adjacency matrices as

$$ A_p=A_{t_l}\cdots{A_{t_2}}{A_{t_1}}$$

* **Graph Transformer Layer** 

<center>
<img class="center medium" src=".//paper-pictures-graph-transformer/figure05.png" width="100%">
</center>

* **Graph Transformer Networks**

<center>
<img class="center medium" src=".//paper-pictures-graph-transformer/figure06.png" width="100%">
</center>

Figure2 demonstrated the overall architecture of the GTN. Firstly, in each channel, the filter will be learnt to assign weight on each types of edges. The details can be seen from the code in their github. 

```python
def forward(self, A, num_nodes, epoch=None, layer=None):
    weight = self.weight
    filter = F.softmax(weight, dim=1)   
    num_channels = filter.shape[0]
    results = []
    for i in range(num_channels):
        for j, (edge_index,edge_value) in enumerate(A):
            if j == 0:
                total_edge_index = edge_index
                total_edge_value = edge_value*filter[i][j]
            else:
                total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
        
        index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=num_nodes, n=num_nodes, op='add')
        results.append((index, value))
    
    return results, filter
```