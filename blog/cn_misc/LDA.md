---
layout: notes
section-type: notes
title: 文本主题模型 LDA
category: cn_misc
---

* TOC
{:toc}
---

本文部分内容摘录自:
* 简书博客[共轭先验、共轭分布——为LDA做准备](https://www.jianshu.com/p/bb7bce40a15a)
* 概率论与数理统计（李书刚 编 湖北科学技术出版社）
* CSDN博客[LDA基础知识系列1](https://blog.csdn.net/m0_37788308/article/details/78935021)
* CSDN博客[LDA基础知识系列2](https://blog.csdn.net/m0_37788308/article/details/78942279)
* CNBlogs[MCMC 蒙特卡罗方法](https://www.cnblogs.com/pinard/p/6625739.html)


## 基本概率分布
<hr>

* 先验分布 prior probability
* 后验分布 posterior probability
* 似然函数 likelihood function
* 共轭分布 conjugacy （后验概率分布函数与先验概率分布函数形式相同）

> 那么对于抛硬币这个事件来说，抛出正面硬币的概率就应该是一个概率的概率，也就是说它的结果不是一个单一的值 1/2，而是一个概率分布，可能有很高的概率是1/2，但是也有一定的概率是100%（比如抛100次结果还真都100次都是正面）。那么在这里这个概率的分布用函数来表示就是一个似然函数，所以似然函数也被称为“分布的分布”。用公式来表示就是：后验概率∝ 似然函数*先验概率

>共轭是指的先验分布(prior probability distribution)和似然函数(likelihood function)。如果某个随机变量Θ的后验概率$P(A\|B)$和先验概率$P(A)$属于同一个分布簇的，那么称$P(A\|B)$和$P(A)$为共轭分布，同时，也称$P(A)$为似然函数$P(B\|A)$的共轭先验。

<br>
<br>
<br>

## 采用共轭先验的原因
<hr>

可以使得先验分布和后验分布的形式相同，这样一方面合符人的直观（它们应该是相同形式的）另外一方面是可以形成一个先验链，即现在的后验分布可以作为下一次计算的先验分布，如果形式相同，就可以形成一个链条。

### 问题的关键
如何选取一个先验分布可以使得后验分布与先验分布有相同的数学形式。


### 贝叶斯公式的一些理解
$$P(B_i|A)=\frac{P(B_i)P(A|B_i)}{\sum_{j=1}^n{P(B_j)P(A|B_j)}},\quad i=1,2,...,n$$

现通过一个具体问题，来说明这个公式的实际意义，若有以病人高烧到40摄氏度，医生要确定他患有何种疾病，首先要考虑病人可能发生的疾病$B_1,\ B_2,\ ...,\ B_n.$ 这里假设一个病人不会同时得几种疾病，即事件 $B_1,\ B_2,\ ...,\ B_n$ 互不相容，医生可凭借以往的经验估计出发病率 $P(B_i),\ i=1,2,...n,$ 这通常叫先验概率。进一步要考虑的是一个人得$B_i$这种病时烧到40摄氏度的可能性$P(A\|B_i),\ i=1,2,...,n,$这可以根据医学知识来确定。这样，就可以根据贝叶斯公式算得$P(B_i\|A),\ i=1,2,...,n.$这个概率表示在已获得新的信息（即知此病人高烧到40摄氏度）后，病人得$B_1,\ B_2,\ ...,\ B_n$这些疾病的可能性大小，这通常称为后验概率，对于较大的$P(B_i\|A)$的$B_i$，为医生的诊断提供了重要依据。

那么在对应的贝叶斯公式中有：

$$P(A), 先验分布$$

$$P(B|A), 似然函数$$

$$P(A|B), 后验分布$$  

则它们之间的关系可以使用贝叶斯公式链接：

$$后验分布=似然函数*先验分布/P(B)$$
即：

$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}∝P(B|A)P(A)$$

这里，如果$P(A\|B)$和$P(A)$拥有相同的数学形式，则$P(A)$是似然函数$P(B\|A)$的共轭先验。

<br>
<br>
<br>

## Beta分布是二项分布和伯努利分布的共轭先验分布
<hr>

证明：  
1.二项分布（似然函数），$n$为样本个数，$k$为概率$A$对应事件所发生的次数，二项分布的似然函数：

$$P(B|A)=C_n^ka^k(1-a)^{n-k} \tag{1}$$  

2.先验分布（Beta分布）。现在假设二项分布的先验分布为Beta分布，超参数为$\alpha,\ \beta$ :  

$$P(A)=P(A|\alpha,\ \beta)=\frac{1}{B(\alpha,\ \beta)}a^{\alpha-1}(1-a)^{\beta-1} \tag{2}$$
注意：$P(A\|\alpha,\ \beta)$不是条件概率，其表示的是$P(A)$的超参数为$\alpha,\ \beta$。  

3.计算后验分布：

$$
\begin{aligned}
P(A|B)&=\frac{P(A)P(B|A)}{P(B)}∝P(A)P(B|A)\\
&=C_n^ka^k(1-a)^{n-k}\ \frac{1}{B(\alpha,\ \beta)}a^{\alpha-1}(1-a)^{\beta-1}\\
&=\frac{C_n^k}{B(\alpha,\ \beta)}a^{(\alpha+k)-1}(1-a)^{(n-k+\beta)-1}  
\end{aligned}
$$  

计算解析：在给定$\alpha,\ \beta$的情况下，$B(\alpha,\ \beta)$是一个常数，观察第三个等号后面的式子，对比Beta分布的概率密度函数：

$$P(A)=f(a)=\frac{1}{B(\alpha,\ \beta)}a^{\alpha-1}(1-a)^{\beta-1} \tag{4}$$
观察系数会发现，可根据第三行参数对应的系数进行配凑Beta分布形式。从而$(3)$式和$(4)$式在形式上是相同的，所以由二项分布似然函数和Beta先验生成的后验分布是Beta分布。  
由此我们可以得出结论，Beta分布是二项分布的共轭先验分布。

<br>
<br>
<br>

## Dirichlet分布是Multinomial分布的共轭先验分布
<hr>

下面的工作就是推广：  
* 二项分布$\rightarrow$多项分布
* Beta分布$\rightarrow$Dirichlet分布

Beta分布概率密度函数：

$$\frac{1}{B(\alpha_1,\ \alpha_2)}{P_1}^{\alpha_1-1}{P_2}^{\alpha_2-1},\ (P_1+P_2=1)$$

Dirichlet分布概率密度函数：

$$\frac{1}{B(\alpha_1,\ \alpha_2,\ \cdots,\ \alpha_n)}{P_1}^{\alpha_1-1}{P_2}^{\alpha_2-1}{P_3}^{\alpha_3-1}\cdots{P_k}^{\alpha_k-1},$$  

$$(P_1+P_2+P_3+\cdots+P_k=1)$$

对于上述两种函数的正式表达形式有： 

Beta分布：

$$f(x)=\begin{cases}
\frac{1}{B(\alpha,\ \beta)}x^{\alpha-1}(1-x)^{\beta-1} &,\ x\in[0,\ 1]\\
0 &,\ other
\end{cases}$$  

其中，$B(\alpha,\ \beta)=\int_0^1x^{\alpha-1}{(1-x)^{\beta-1}}dx=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$
  
Dirichlet分布：

$$f(\vec{p}|\vec{\alpha})=\begin{cases}
\frac{1}{\Delta(\vec{\alpha})}\prod_{k=1}^{K}p_k^{\alpha_k-1} &,\ p_k\in[0,\ 1]\\
0 &,\ other
\end{cases}$$  

简记：$Dir(\vec{p}\|\vec{\alpha})=\frac{1}{\Delta(\vec{\alpha})}\prod_{k=1}^{K}p_k^{\alpha_k-1}$，其中$\Delta_k(\vec{\alpha})=\frac{\prod_{k=1}^{K}\Gamma(\alpha_k)}{\Gamma(\sum_{k=1}^{K}\alpha_k)},\ \sum_{k=1}^{K}p_k=1$  
其中，$\alpha$是向量参数，共$K$个，定义在$x_1,\ x_2,\cdots,\ x_{k-1}$维上，这里$x_1+x_2+\cdots+x_{k-1}=1,\ x_1,\ x_2,\ \cdots, x_{k}>0$  

这里，对于$\alpha$中的少量参数进行人为指定，但是在LDA中会涉及到很多个主题，很多个词，在进行高维的建模时，怎么对这么多$\alpha$进行取值呢？同时，我们也没有更多的先验知识来确定哪一个$\alpha$更重要，于是公平起见：  
所有的$\alpha$参数都使用同一个$\alpha_k$来表征，也就是说把每个组的超参数选作一样，这就是**对称Dirichlet分布**。 

对称Dirichlet分布：  
$Dir(\vec{p}\|\vec{\alpha})=\frac{1}{\Delta(\vec{\alpha})}\prod_{k=1}^Kp_k^{\alpha_k-1}$，其中$\Delta_k(\vec{\alpha})=\frac{\Gamma^K(\alpha_k)}{\Gamma(K\alpha_k)}$  
在此，我们引用维基百科上面关于Dirchilet分布中$K=3$的动图来解释$\alpha$参数对于概率$p$的影响。[动图链接](https://en.wikipedia.org/wiki/File:LogDirichletDensity-alpha_0.3_to_alpha_2.0.gif), 下面为截图:  
* 当$\alpha_1=\alpha_2=\alpha_3=\alpha=1$时，退化为均匀分布：  

<center>
<img src=".\pictures\001.png">
Figure 1
</center>  

* 当$\alpha_1=\alpha_2=\alpha_3=\alpha>1$时，$p_1=p_2=p_3$的概率增大  

<center>
<img src=".\pictures\002.png">
Figure 2
</center>

* 当$\alpha_1=\alpha_2=\alpha_3=\alpha<1$时，$p_i=1$的概率增大,而$p_{k\neq{i}}=0$的概率增大  

<center>
<img src=".\pictures\003.png">
Figure 3
</center>

图像解析：
* 当$\alpha_1=\alpha_2=\alpha_3=\alpha<1$时，$p_i=1$的概率增大,而$p_{k\neq{i}}=0$的概率增大，这其实意味着文档属于某一个主题的概率很大，接近于1，属于其它主题的概率就很小，接近于0，就像Figure 3一样，在坐标轴上的每一个$p_i, 或者是x_i$在取值为1时，其函数值都会变大，而其它参数即$p_{k\neq{i}}$取值则接近于0。
* 当$\alpha_1=\alpha_2=\alpha_3=\alpha>1$时，$p_1=p_2=\cdots=p_k$的概率增大，这意味着文档对于每一个主题的偏好是一样的，都随便，这很不利于主题词找到其心之所属啊！所以，在使用Dirichlet分布的时候，我们将倾向于使用$\alpha<1$的数值。

综上所述，我们使用Dirichlet分布作为多项分布的共轭先验分布，依据对称Dirichlet分布的性质，$\alpha$将取值为小于1的数值，有利于对文档主题进行分类。

<br>
<br>
<br>

## LDA主题模型
<hr>

最简单的就是Unigram Moedel，就是认为上帝是按照如下的游戏规则产生文本的。

* 上帝只有一个骰子，这个骰子有V个面，每个面对应一个词，各个面的概率不一；
* 每抛一次骰子，抛出的面就对应的产生一个词；如果一篇文档中有$n$个词，上帝就是独立的抛$n$次骰子产生这$n$个词；

<center>
<img src=".\pictures\010.png">  
Figure 5
</center>

上帝的这个唯一的骰子各个面的概率记为$\vec{p}=(p_1,p_2,\cdots,p_V)$，所以每次掷骰子类似一个伯努利实验。那么我们把这个抛$V$面骰子的实验记为$w\sim{Mult(w\|\vec{p})}$

但对于上面的描述，贝叶斯统计学派会有不同意见，他们会很挑剔的批评：假设上帝拥有唯一一个固定的骰子是不合理的。在贝叶斯统计学派看来，一切参数都是随机变量，以上模型中的骰子$\vec{p}$不是唯一固定的，它也是一个随机变量。所以按照贝叶斯学派的观点，上帝是按照以下的过程在玩游戏的

* 上帝优异装有无穷多骰子的坛子，里面有各式各样的骰子，每个骰子有$V$个面；
* 上帝从坛子里面抽了一个骰子出来，然后用这个骰子不断的抛，然后产生了预料中所有的词；

上帝的这个坛子里面，骰子可以是无穷多个，有些类型的骰子数量多，有些类型的骰子数量少，所以从概率分布的角度看，坛子里面的骰子$\vec{p}$服从一个概率分布$p(\vec{p})$，这个的分布称为参数$\vec{p}$的先验分布。

<center>
<img src=".\pictures\011.png">  
Figure 6
</center>

前面提到了，多项分布的共轭先验分布就是Dirichlet分布，所以我们就能使用下图来表示这个过程：

<center>
<img src=".\pictures\012.png">  
Figure 6
</center>

最终，我们的Unigram Model可以表示成如下图所示的情形：

<center>
<img src=".\pictures\013.png">  
Figure 7
</center>

<br>
<br>
<br>

## Topic Model 和 PLSA
<hr>

以上Unigram Model 是一个很简单的模型，模型中的假设看起来过于简单，和人类写文章产生每一个词的过程差距比较大，有没有更好的模型呢？

我们可以试想一下日常生活中认识如何构思文章的。如果我们要写一篇文章，往往是先确定要写那几个主题。譬如构思一篇自然语言处理相关的文章，可能40%会谈语言学、30%谈论概率统计、20%谈论计算机、还有10%谈论其它的主题：

* 说到语言学，我们容易想到以下一些词：语法、句子、乔姆斯基、句法分析、主语...
* 谈论概率统计，我们容易想到以下一些词：概率、模型、均值、方差、证明、独立、马尔科夫链、...
* 谈论计算机，我们容易想到的词是：内存、硬盘、编程、二进制、对象、算法、复杂度...

以上这种直观的想法由Hoffmn于1999年给出的PLSA（Probability Latent Semantic Analysis）模型中首先进行了明确的数字化。Hoffmn认为一篇文档（Document）可以由多个主题（Topic）混合而成，而每个Topic都是词汇上的概率分布，文章中的每个词都是由一个固定的Topic生成的。

那么，所有人类思考和写文章的行为都可以认为是上帝的行为，我们继续回到上帝的假设中，那么在PLSA模型中，Hoffmn认为上帝是按照如下的游戏规则来生成文本的

* 上帝有两种类型的骰子，一类是doc-topic骰子，每个doc-topic骰子有$K$个面，每个面有一个topic编号；一类是topic-word骰子，每个topic-word骰子有$V$个面，每个面对应一个词；
* 上帝一共有$K$个topic-word骰子，每个骰子有一个编号，编号从1到$K$；
* 生成每篇文档之前，上帝都先为这篇文章制造一个特定的doc-topic骰子，然后重复如下过程生成文档中的词
    * 投掷这个doc-topic骰子，得到一个topic编号
    * 选择$K$个topic-word骰子中编号为$z$的那个，投掷这个骰子，于是得到一个词

<center>
<img src=".\pictures\014.png">  
Figure 7
</center>

我们可以发现在以上的游戏规则下，文档和文档之间是独立可交换的，同一个文档内的词也是独立可交换的，还是一个bag-of-words模型。游戏中的$K$个topic-word骰子，我们可以记为$\vec{\varphi_1},\cdots,\vec{\varphi_K}$，对于包含$M$篇文档的语料$C=(d_1,d_2,\cdots,d_M)$中的没篇文档$d_m$，都会有一个特定的doc-topic骰子$\vec{\theta_m}$，所有对应的骰子应记为$\vec{\theta_1},\cdots,\vec{\theta_M}$.于是在PLSA这个模型中，第m篇文档$d_m$中的每个词的生成概率为:

$$p(w|d_m)=\sum_{z=1}^{K}p(w|z)p(z|d_m)=\sum_{z=1}^{K}{\varphi_{zw}\theta_{mz}}$$

所以整篇文档的生成概率为：

$$p(\vec{w}|d_m)=\prod_{i=1}^{n}\sum_{z=1}^{K}p(w_i|z)p(z|d_m)=\prod_{i=1}^{n}\sum_{z=1}^{K}{\varphi_{zw_i}\theta_{dz}}$$

<br>
<br>
<br>

## LDA文本建模
<hr>

对于上述的PLSA模型，贝叶斯学派显然是有意见的，doc-topic骰子$\vec{\theta_m}$和topic-word骰子$\vec{\varphi_k}$都是模型中的参数，参数都是随机变量，怎么能没有先验分布呢？于是，类似于对Unigram Model的贝叶斯改造，我们也可以如下在两个骰子参数前加上先验分布从而把PLSA对应的游戏过程改造成一个贝叶斯的游戏过程。由于$\vec{\varphi_k}$和$\vec{\theta_m}$都对应多项分布，所以先验分布的一个好的选择就是Dirichlet分布，于是我们就得到了LDA（Latent Dirichlet Allocation）模型。

<center>
<img src=".\pictures\015.png">  
Figure 8
</center>

在LDA模型中，上帝是按照如下的规则进行的：

* 上帝有两大坛子的骰子，第一个摊子装的是doc-topic骰子，第二个坛子装的是topic-word骰子；

<center>
<img src=".\pictures\016.png">  
Figure 9
</center>

* 上帝随机的从第二个坛子中独立的抽取了$K$个topic-word骰子，编号为1到$K$；
* 每次生成一篇新的文档前，上帝先从第一个坛子中随机抽取一个doc-topic骰子，然后重复如下过程生成分文档中的词
    * 投掷这个doc-topic骰子，得到一个topic编号$z$
    * 选择$K$个topic-word骰子中编号为$z$的那个，投掷这个骰子，于是得到一个词


使用LDA概率模型图表示，LDA模型的游戏过程如图所示：

<center>
<img src=".\pictures\004.png">
Figure 4
</center>

该概率模型分为两个阶段：
* $\vec{\alpha}\rightarrow\vec{\theta_m}\rightarrow{Z_{m,\ n}}$：选取一个参数为$\vec{\theta_m}$的doc-topic分布，然后对第$m$篇文章的第$n$个词的topic，生成$Z_{m,\ n}$的编号。

* $\vec{\beta}\rightarrow\vec{\psi_k}\rightarrow{W_{m,\ n\|k=Z_{m,\ n}}}$：生成第$m$篇文章的第$n$个词。

由概率图可以看出，一篇文章m的生成过程为:  
* 从一个参数为$\alpha$的Dirichlet分布中采样出一个multinomial分布$\theta_m$，作为该文章在$k$个主题上的分布  
* 对该文章里的每个词$n$，根据上步中的$\theta_m$分布，采样出一个topic编号来，然后根据此topic-word对应的参数为$\psi$的多项分布，采样出一个词。

总结：我们从Unigram得到词和主题生成概率，然后结合这个结论应用到LDA中去，最后我们就可以得到每篇文章的生成概率。


### LDA in Text
LDA在主题建模中的应用，需要知道以下几点：
* 文档集中的words不考虑顺序，符合词袋模式，假设总词汇数为$V$
* 每篇由$n$个word生成的Document，每个word的生成都服从多项分布，就像上帝拥有一个$V$面的骰子（每面对应一个word），抛$n$次就可以生成一篇Document了。
* Document与Document之间的骰子不是同一个，每次为Document选一个Topic骰子，这个过程也服从多项分布。

一个通俗的例子如下：
* 一个作家写了$m$篇文章，一共涉及了$K$个Topic，每个Topic下的词分布为一个服从参数为$\beta$的Dirichelet先验分布中采样出来的多项分布。
* 对于每篇文章，他会从一个泊松分布采样出一个值作为文章长度，然后从一个服从参数为$\alpha$的Dirichlet先验分布中采样出一个多项分布作为改文章里面每个Topic下出现的词的概率。
* 也就是说，当作家想写第$m$篇文章的第$n$个词的时候，他是通过两个多项分布决定的。不断重复这样的随机生成过程，直到他把$m$篇文章都写完。


### LDA的参数推导

从上面的过程中可知，参数$\alpha$和$\beta$是由经验给出的，$Z_{m,\ n},\ \theta_m,\ \psi_k$是隐含变量，需要推导。LDA的推导方法有两种，一种是精确推导，需要使用EM算法计算，另一种是近似推导，实际工程中通常用这种方法，其中最简单的方法就是Gibbs采样法。

**Gibbs采样**：是[MCMC](https://www.cnblogs.com/pinard/p/6625739.html)的一种，其主要思想就是每次迭代只改变一个维度的值，从而逼近得到我们所期待的目标函数概率分布。

其中过程详见文章MCMC，里面解释了采样方法对目标函数概率分布的暴力求解方法。

那么，我们根据LDA可以生成联合分布$p(\vec{\bf w},\vec{{\bf z}})$，这里我们根据观察，已经得到了文章中的词汇分布$\vec{{\bf w}}$，那么Gibbs采样的主要参数估计就是针对$\vec{{\bf z}}$这个隐含的变量，也就是这些文章中主题分布的概率，有$\vec{{\bf z}}=(\vec{z_1},\vec{z_2},\cdots,\vec{z_M})$这里的$M$表示的是一共有M篇文档，所以接下来的任务就是求解每篇文档中，各个主题出现的概率问题，如下图所示：

<center>
<img src=".\pictures\017.png">
Figure 10
</center>

我们根据实际观察到的每篇文章中的词汇分布$\vec{w}$从而使用Gibbs采样得到$\vec{z}$，因为Gibbs采样在之前的文档中说过可以每次只更改一个变量从而求得一个分布，最后我们就可以得到$\varphi_{kt}$，然后我们个亿统计每篇文档中的topic频率分布，我们就可以计算出和$\theta_{mk}$。

那么具体训练的流程如下：

| Algotithm1 LDA Training 算法 | 
| :------ |
|1：随机初始化：对语料中每篇文档中的每个词$w$，随机的赋一个topic编号$z$ |
|2: 重新扫描语料库，对每个词$w$，按照Gibbs Sampling公式重新采样它的topic，在语料中进行更新； |
|3：重复以上语料库的重新采样过程直到Gibbs Sampling收敛 |
|4: 统计语料库的topic-word贡献频率矩阵，该矩阵就是LDA的模型 |









