---
layout: notes
section-type: notes
title: Convex Optimization
category: math
---

* TOC
{:toc}
---

## Previous Lectures
* Optimization Model and Software Tool (Instructor: Qingxing Dong)

| | | | | |
|---|---|---|---|---|
|[Lecture1](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture1ConvexSets-handouts.pdf)|[Lecture2](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture2ConvexFunctions-handouts.pdf)|[Lecture3](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture3ConvexSets2-handouts.pdf)|[Lecture4](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture4ConvexSets3-handouts.pdf)|[Lecture5](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture5DescentAlgorithm-handouts.pdf) |
|[Lecture6](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture6DescentAlgorithm2-handouts.pdf)|[Lecture7](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture7-Gradient-Method-handouts.pdf)|[Lecture8](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture8-QuasiNewton-handouts.pdf)|[Lecture9](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture9-LagrangeDuality-handouts.pdf)|[Lecture10](https://heming-zhang.github.io/course/IntroConvexOptimization/Lecture10-KKT-Condition-handouts.pdf)|

* Self-Recommended Books
  * [Convex Optimization(Boyd)](https://heming-zhang.github.io/course/ConvexOptimization(CN).pdf)
  * [Methods of Optimization](https://heming-zhang.github.io/course/Methods_of_Optimization(2015).pdf)
  * [Numerical Method for Solving Systems of Nonlinear Equations](https://www.lakeheadu.ca/sites/default/files/uploads/77/docs/RemaniFinal.pdf)


## Similar Courses
* [WashU ESE 415 Convex Optimization(2019 Spring)](https://cigroup.wustl.edu/teaching/ese415-2019/)

## 1 Basic Concepts

## 2 Unstricted Optimization Methods

### 2.1 Gradient Descent Algorithm
> (1) Given a function $f(\mathbf{x})$  
> (2) Intialize the start point $\mathbf{x}^{(0)}$, $k\leftarrow{0}$  
> (3) Judge whether $\mathbf{x}^{(k)}$ is the minimum or approximate minimum. If so, stop algorithm, and if not, jump to next step  
> (4) Use negative gradient $-\nabla_{\mathbf{x}}{f}(\mathbf{x}^{(k)})^{\text{T}}$ as search direction $\mathbf{p}^{(k)}$(fastest descent direction)  
> (5) Use **Steepest Descent(Line Search)** to find $\lambda_k=\arg\ \min_{\lambda_{k}}{f(\mathbf{x}^{(k)}+\lambda_k{\mathbf{p}^{(k)}})}=\arg\ \min_{\lambda_{k}}{\phi(f(\mathbf{x}^{(k)},\lambda_k))}$  
> OR (5') Use **Approximate Optimal Stepsize** to find $\lambda_k$
>
> * If $f(\mathbf{x})$ has first-order continuous partial derivative, and use **Taylor First-Order Expansion** at point $\mathbf{x}^{(k)}$ with negative gradient $-\nabla_{\mathbf{x}}{f}(\mathbf{x}^{(k)})^{\text{T}}$ as search direction
>
> $$
\begin{aligned}
f(\mathbf{x}^{(k)}+\lambda_k{\mathbf{p}^{(k)}}) &=f(\mathbf{x}^{(k)}-\lambda_k{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}})\\
&=f(\mathbf{x}^{(k)})-{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}}\big[\lambda_k{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})}\big]
\end{aligned}$$
>
> * And sometimes, we regulate $\mathbf{p}^{(k)}=\frac{-\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}}{\|\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}\|}$, and we get
>
> $$\lambda_{k}=\lambda_{0}||\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})||$$
> 
> * Finally, we get
> 
> $$
\begin{aligned}
\mathbf{x}^{(k+1)}&=\mathbf{x}^{(k)}+\lambda_k{\mathbf{p}^{(k)}}\\
&=\mathbf{x}^{(k)}+(\lambda_{0}||\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})||)(-\frac{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})}{||\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})||})\\
&= \mathbf{x}^{(k)}-\lambda_0{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})}
\end{aligned}
$$
>
> * If $f(\mathbf{x})$ has second-order continuous partial derivative, and use **Taylor Second-Order Expansion** at point $\mathbf{x}^{(k)}$ with negative gradient $-\nabla_{\mathbf{x}}{f}(\mathbf{x}^{(k)})^{\text{T}}$ as search direction
>
> $$
\begin{aligned}
f(\mathbf{x}^{(k)}+\lambda_k{\mathbf{p}^{(k)}}) &=f(\mathbf{x}^{(k)}-\lambda_k{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}})\\
&=f(\mathbf{x}^{(k)})-{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}}\big[\lambda_k{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})}\big]\\&+\frac{1}{2}\big[\lambda_k{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}}\big]\mathbf{H}(\mathbf{x}^{(k)})\big[\lambda_k{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}}\big]
\end{aligned}$$
>
> * Hence, we get the best stepsize by taking derivative for $\lambda_k$
>
> $$\lambda_k=\frac{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})}{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}{\mathbf{H}(\mathbf{x}^{(k)})}\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})}$$
>
> * And sometimes, we regulate $\mathbf{p}^{(k)}=\frac{-\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}}{\|\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}\|}$, and we get
>
> $$\lambda_k=\frac{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})\|\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}\|}{\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})^{\text{T}}{\mathbf{H}(\mathbf{x}^{(k)})}\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})}$$

### 2.2 Newton's Method
> If $f(\mathbf{x})$ has continuous second-order partial derivative, and $\mathbf{x}^{(k)}$ as its approximate minimum, which we have second-order Taylor Expansion:
>
> $$f(\mathbf{x})\approx{ f(\mathbf{x}^{(k)})+\nabla{f(\mathbf{x}^{(k)})}^{\text{T}}(\mathbf{x}-\mathbf{x}^{(k)})+\frac{1}{2}{(\mathbf{x}-\mathbf{x}^{(k)})^{\text{T}}}\mathbf{H}(\mathbf{x}^{(k)})(\mathbf{x}-\mathbf{x}^{(k)})}$$
>
> Take $(\mathbf{x}-\mathbf{x}^{(k)})$ as main variable to take derivative for above equation, and we get
>
> $$\mathbf{p}^{(k)}=(\mathbf{x}-\mathbf{x}^{(k)})=-\mathbf{H}(\mathbf{x}^{(k)})^{-1}\nabla{f(\mathbf{x}^{(k)})}$$
>
> Hence we usually call $\mathbf{p}^{(k)}=-\mathbf{H}(\mathbf{x}^{(k)})^{-1}\nabla{f(\mathbf{x}^{(k)})}$ as **Newton Direction**.

> And we can continue to find **Approximate Optimal Stepsize** with $\lambda_k=\arg\ \min_{\lambda_{k}}{f(\mathbf{x}^{(k)}+\lambda_k{\mathbf{p}^{(k)}})}=\arg\ \min_{\lambda_{k}}{f(\mathbf{x}^{(k)}-\lambda_k{\mathbf{H}(\mathbf{x}^{(k)})^{-1}\nabla{f(\mathbf{x}^{(k)})}})}$  

### 2.3 Quasi-Newton method
Since it is hard to compute **Hessian Matrix** if variable $\mathbf{x}$ has high dimensions.

* DFP Method: Approximate Inverse of Hessian Matrix with $\bar{\mathbf{H}}^{(k)}$
* BFGS Method: Approxiamte Hessian Matrix ${\mathbf{B}}^{(k)}$