---
layout: notes
section-type: notes
title: Advanced Algorithms
category: cs
---

* TOC
{:toc}
---

Recommended Books

* [Introduction to Algorithms(2009)(3rd Edition)](https://heming-zhang.github.io/course/Introduction_to_Algorithms(2009)(3rd_Edition).pdf)


## Note 1 Running Time
### Pseudo-Polynomial Time
> By the definition, an algorithm runs in pseudo-polynomial time if the running time of is a polynomial, when the input is in numeric value, but this is not neccessarily true when we consider the input length in terms bits representation which would result in another running time. 

### P1: The "Fill-a-Bag" Problem:
Like the 0,1-KnapSack Problem, we can use the table to solve this.


## Note 2 Division And Conquer
### Recurrences
* MergeSort
* Maximum Subarray

### Solving the recurrences
* Substitution Method
* Recursion Tree, $T(n)=T(n/3)+T(2n/3)+\Theta(n)$
* Master Method, Only when problem are all the same size

### P1: Binary Search
```Python
while(low<high):
    mid = (low + high) / 2
    if A[mid] == x: return mid
    else if A[mid] > x:
        high = mid
    else
        low = mid + 1
```

$T(n)=T(n/2)+\Theta(1)$

### P2: Insertion Sort
* [A useful chinese blog link](https://www.cnblogs.com/youxin/archive/2012/03/09/2387426.html)
```c
void Insert(int *a,int n)
{
    int i=n-1;
    int key=a[n];
    while((i>=0)&&(key<a[i]))
    {
        a[i+1]=a[i];
        i--;
    }
    a[i+1]=key;
    return;
}
```

```c
void InsertionSort(int *a,int n)
{
    if(n>0)
    {
        InsertionSort(a,n-1);
        Insert(a,n);
    }
    else 
        return;
}
```

Time complexity: $T(n)=T(n-1)+\Theta(n)$


### P3: Integer Multiplication
Let $a_1$, $a_2$ denote the leftmost and rightmost half the digits of an $n$-digit positive integer $a$, the same for $b$. In this way, we can get

$a\times{b}=(a_1\times{}10^{n/2}+a_2)(b_1\times{}10^{n/2}+b_2)=a_1b_1\times{10^n}+(a_1b_2+a_2b_1)\times{10^{n/2}}+a_2b_2$

If we calcualte all of them, we need $T(n)=4T(n/2)+\Theta(n)$

But if we only calculate three of them with 
* $a_1b_1$
* $a_2b_2$
* $a_1b_2+a_2b_1=(a_1+a_2)(b_1+b_2)-a_1b_1-a_2b_2$

Then we only need $T(n)=3T(n/2)+\Theta(n)$

Until now, we reduce this problem to $\Theta(n\log{n})$

### P4: Selection Problem
* Quick Sort: [Chinese Blog Link](https://wiki.jikexueyuan.com/project/easy-learn-algorithm/fast-sort.html)

* Quick Selection