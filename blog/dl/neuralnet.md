---
layout: notes
section-type: notes
title: Machine Learning with Nets
category: ml
---

* TOC
{:toc}
---
* History Papers
    * A new model learning in graph domains

* Course Material
    * Course [CS 231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/syllabus.html)
    * Course [Video Link](https://www.youtube.com/results?search_query=cs+231n)


## 1. Convolutional Neural Networks
### 1.1 Convolute
<center>
<img class="center large" src=".//net/net001.png" height="50%" width="70%">
</center>

* Dynamic [Demo Link](https://cs231n.github.io/convolutional-networks/) for Convolution 

* Also, Conv1D and Conv2D have their own application.

<center>
<img class="center large" src=".//net/net034.png" height="50%" width="70%">
</center>

Above picture show the application of Conv1D in NLP. By using Conv1D with different kernel sizes, the matrix will be transformed into multiple column vectors.

<center>
<img class="center large" src=".//net/net002.png" height="50%" width="70%">
</center>

While Conv2D will keep transform the original matrix to 2D matrices by multiple filters with concatentation.

### 1.2 Pooling
<center>
<img class="center large" src=".//net/net003.png" height="50%" width="70%">
</center>

### 1.3 Activation Functions
<center>
<img class="center large" src=".//net/net004.png" height="50%" width="70%">
</center>

<center>
<img class="center large" src=".//net/net005.png" height="50%" width="70%">
</center>

### 1.4 Weight Initialization
* Xavier Initialization

### 1.5 Batch Normalization
<center>
<img class="center large" src=".//net/net006.png" height="50%" width="80%">
</center>

### 1.6 Transfer Learning
* [Demo Link](https://blog.csdn.net/SunshineSki/article/details/84086760)

### 1.7 Optimizer
* SGD + Momentum
<center>
<img class="center large" src=".//net/net007.png" height="50%" width="80%">
</center>

* Nesterov Momentum
<center>
<img class="center large" src=".//net/net008.png" height="50%" width="80%">
</center>

* Adam
<center>
<img class="center large" src=".//net/net009.png" height="50%" width="80%">
</center>

### 1.8 Learning Rate Decay
<center>
<img class="center large" src=".//net/net010.png" height="50%" width="80%">
</center>

### 1.9 Regularization
* Dropout
* [Dropout Lecture Link](http://cs231n.stanford.edu/slides/2020/lecture_8.pdf)


## 2. CNN Architectures
### 2.1 AlexNet
<center>
<img class="center large" src=".//net/net011.png" height="50%" width="80%">
</center>

### 2.2 VGGNet
<center>
<img class="center large" src=".//net/net012.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net013.png" height="50%" width="80%">
</center>

### 2.3 GoogLeNet
<center>
<img class="center large" src=".//net/net014.png" height="50%" width="80%">
</center>

* Due to huge computation demands, using bottlenect filter can reduce amount of data.
<center>
<img class="center large" src=".//net/net015.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net016.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net017.png" height="50%" width="80%">
</center>

### 2.4 ResNet
<center>
<img class="center large" src=".//net/net018.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net019.png" height="50%" width="80%">
</center>

## 3. RNN (Recurrent Neural Networks)
### 3.1 Genreal Applications
<center>
<img class="center large" src=".//net/net020.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net021.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net022.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net023.png" height="50%" width="80%">
</center>

### 3.2 Recurrence Formula
<center>
<img class="center large" src=".//net/net024.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net031.png" height="50%" width="80%">
</center>

[Sorry For Chinese]
<center>
<img class="center large" src=".//net/net030.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net028.png" height="50%" width="80%">
</center>

### 3.3 Recurrent Model
<center>
<img class="center large" src=".//net/net025.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net026.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net027.png" height="50%" width="80%">
</center>

### 3.4 Model Applications
<center>
<img class="center large" src=".//net/net029.png" height="50%" width="80%">
</center>

### 3.5 LSTM (Long Short Term Memory)
<center>
<img class="center large" src=".//net/net032.png" height="50%" width="80%">
</center>

<center>
<img class="center large" src=".//net/net033.png" height="50%" width="80%">
</center>