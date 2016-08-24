---
published: true
layout: post
title:  "Expectation Maximization"
categories: math
tag: EM
---
I try to come up with a short summary for EM. The majority of this summary comes from Chapter 9 in [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/#prml-book). Basically, you can read right ending at section `details`. I enclosed a detailed (or overly long) example of mixture Gaussian in the section. You could just skim for the results if the math hurts. The idea behind EM is very clear but when it comes to pratical models, the derivation would be tedious. I guess that's why people (including me) have learnt EM more than once but may hardly link up the gap in-between. For recaping readers, you may go through all the theory parts to see whether the whole system makes any sense.

As the whole summary is still based on my own understanding, if there's any mistake, feel free to contact me :)

## General EM

### Objective Function
__Definition:__ Denote $$\theta$$ the parameter for the distribution, $$X$$ the observed variable and $$Z$$ the latent variable (for complete probability). The objective function we what to maximize w.r.t. $$\theta$$ is the log likelihood of the observed part:

$$\ln p(X\vert \theta).$$

### Why EM works
The important observation in EM is that $$\ln P(X\vert \theta)$$ can be decomposed as

$$ \begin{align}
\ln p(X\vert \theta)&=\sum_Z q(Z)\ln \frac{p(X,Z\vert \theta)}{q(Z)}+\sum_Z q(Z)\ln\frac{q(Z)}{p(Z\vert X,\theta)}\\
&=L(q,\theta) + DL(q\|p).
\end{align} $$

Here we denote $$L(q,\theta)$$ to be the first term and $$DL(q\|p)$$ to be the second one.

Note that $$DL(q\|p)$$ is always equal to or greater than $$0$$. This shows $$L(q,\theta)$$ is always a lower bound for $$\ln p(X\vert \theta)$$.

Therefore, the `E step` and `M step` could be regarded as an increment for the lower bound $$L(q,\theta)$$ by either fixing $$\theta$$ or $$q$$.

To be specific,

* `E step`, fix $$\theta$$ as $$\theta^{old}$$, $$q=\arg\max_{q}L(q,\theta^{old})$$
* `M step`, fix $$q$$ as $$q^{old}$$, $$\theta=\arg\max_{\theta}L(q^{old},\theta)$$

## Details and Example
### Details
Usually, the `E step` is easy. Notice that in `E step`, when we fix $$\theta=\theta^{old}$$, $$p(Z\vert X,\theta^{old})$$ is irrelevant of $$q$$. Besides, we still have

$$ \begin{align}
\ln p(X\vert \theta^{old})&=L(q,\theta^{old}) + DL(q\|p)\\
&\geq L(q,\theta^{old}).
\end{align} $$

Namely, the maximal value could be obtained iff $$q=p$$ ($$q(Z)=p(Z\vert X,\theta^{old})$$).

For the `M step`, we could get rid of some constants for less computation:

$$ \begin{align}
L(q^{old},\theta) &= \sum_Z q^{old}(Z)\ln \frac{p(X,Z\vert \theta)}{q^{old}(Z)}\\
&= \sum_Z q^{old}(Z)\ln p(X,Z\vert \theta)+H(q^{old})\\
&= \sum_Z q^{old}(Z)\ln p(X,Z\vert \theta)+\text{constant}
\\
&= \sum_Z p(Z\vert X,\theta^{old})\ln p(X,Z\vert \theta)+\text{constant}\\
&= \mathbb{E}_{Z\vert X,\theta^{old}}\ln p(X,Z\vert \theta)+\text{constant}.
\end{align} $$

Here $$H(q)=\sum_Z q(Z) \ln \frac{1}{q(Z)}$$ is the entropy function. And if we let

$$Q(\theta\vert \theta^{old})=\mathbb{E}_{Z\vert X,\theta^{old}}\ln p(X,Z\vert \theta),$$

we see this is the typical objective function that textbooks try to maximize during the `M step`.

By combining the discussions above, we get a clear rewrite of EM process as follows:

* `E step`, calculate $$p(Z\vert X,\theta^{old})$$
* `M step`, compute new $$\theta$$ as $$\theta=\arg\max_{\theta}Q(\theta\vert \theta^{old})$$

### Example
Let's get into a real-world model: mixture Gaussian. In this setting, we assume data are drawn from a weighted bunch of Gaussians:

$$p(X)=\sum_k \pi_k\mathcal{N}(X\vert \mu_k, \sigma_k^2).$$

The first thing we need to do is introducing latent variable $$Z$$ s.t. $$p(X)=\sum_Z p(X,Z)$$ is valid. We could specify $$p(Z)$$ and $$p(X\vert Z)$$ for this purpose. Actually, we can view each data point $$X$$ as being generated through a two-step cluster-like process:

* Choose one component (Gaussian) of the model through a categorical distribution (addressed by $$Z$$)
* Generate $$X$$ based on this single Gaussian

![Alt text](http://g.gravizo.com/g?
  digraph G {
    Z -> X;
  }
)

The above lines are equivalent in saying that

$$\begin{align}
p(Z=k)&=\pi_k\\
p(X\vert Z=k)&=\mathcal{N}(X\vert \mu_k, \sigma_k^2),
\end{align}$$

which can be verified is consistent in getting the same marginal of $$p(X)$$.

Nevertheless, introducing $$Z$$ is not always as simple as we do here. However, $$Z$$ is very intuitive given the *__graphical models__* structures.

Another important feature in simplifying the calculation is the i.i.d. property. Suppose $$X=\{X_i\}_i$$ and $$Z=\{Z_i\}_i$$. Actually, as the *__i.i.d.__* property suggests,

$$ \begin{align}
p(Z\vert X,\theta^{old})&=\frac{p(X,Z\vert \theta^{old})}{\sum_Z p(X,Z\vert \theta^{old})}\\
&=\frac{\prod_i p(X_i,Z_i\vert \theta^{old})}{\sum_Z\prod_i p(X_i,Z_i\vert \theta^{old})}\\
&=\frac{\prod_i p(X_i,Z_i\vert \theta^{old})}{\prod_i \sum_{Z}p(X_i,Z_i\vert \theta^{old})}\\
&=\frac{\prod_i p(X_i,Z_i\vert \theta^{old})}{\prod_i \sum_{Z_i} p(X_i,Z_i\vert \theta^{old})}\\
&=\frac{\prod_i p(X_i,Z_i\vert \theta^{old})}{\prod_i  p(X_i\vert \theta^{old})}\\
&=\prod_i p(Z_i\vert X_i,\theta^{old}).
\end{align} $$

Namely, in `E step`, we only need to calculate each $$p(Z_i\vert X_i,\theta^{old})$$ without taking the variables as a whole.

Similarly, in `M step`,

$$ \begin{align}
Q(\theta\vert \theta^{old})&=\mathbb{E}_{Z\vert X,\theta^{old}}\ln p(X,Z\vert \theta)\\
&=\mathbb{E}_{Z\vert X,\theta^{old}}\sum_i \ln p(X_i,Z_i\vert \theta)\\
&=\sum_i\mathbb{E}_{Z\vert X,\theta^{old}} \ln p(X_i,Z_i\vert \theta)\\
&=\sum_i \mathbb{E}_{Z_i\vert X_i,\theta^{old}} \ln p(X_i,Z_i\vert \theta).
\end{align} $$

The third equality comes from the summation property of expectation. The last line holds since $$Z_i\vert X_i,\theta^{old}$$ are independent as we have proved above.

<!--
Similarly, in `M step`,
$$ \begin{align}
Q(\theta\vert \theta^{old})&=\mathbb{E}_{Z\vert X,\theta^{old}}\ln p(X,Z\vert \theta)\\
&=\mathbb{E}_{Z\vert X,\theta^{old}}\sum_i \ln p(X_i,Z_i\vert \theta)\\
&=\sum_Z p(Z\vert X,\theta^{old})\sum_i \ln p(X_i,Z_i\vert \theta)\\
&=\sum_Z\sum_i p(Z\vert X,\theta^{old}) \ln p(X_i,Z_i\vert \theta)\\
&=\sum_Z\sum_i\prod_j p(Z_j\vert X_j,\theta^{old}) \ln p(X_i,Z_i\vert \theta)\\
&=\sum_i\sum_Z\ln p(X_i,Z_i\vert \theta)\prod_j p(Z_j\vert X_j,\theta^{old}) \\
&=\sum_i\sum_{Z_i}p(X_i,Z_i\vert \theta)\ln p(X_i,Z_i\vert \theta)\prod_{j\neq i} \sum_{Z\backslash \{Z_i\}}p(Z_j\vert X_j,\theta^{old}) \\
&=\sum_i\sum_{Z_i}p(X_i,Z_i\vert \theta)\ln p(X_i,Z_i\vert \theta)\\
&=\sum_i \mathbb{E}_{Z_i\vert X_i,\theta^{old}} \ln p(X_i,Z_i\vert \theta)
\end{align} $$
-->

Now, we can pin down the formulas for `E step` and `M step`.

* `E step`, we need to derive $$p(Z_i\vert X_i,\theta^{old})$$.
Notice $$\theta=\{\pi_k,\mu_k,\sigma_k^2\}_k$$. This conditional probability can be given by the Bayes formula:

$$ \begin{align}
p(Z_i=k\vert X_i,\theta^{old})&=\frac{p(X_i\vert Z_i=k,\theta^{old})p(Z_i=k\vert \theta^{old})}{\sum_{k'} p(X_i\vert Z_i=k',\theta^{old})p(Z_i=k'\vert \theta^{old})}\\
&=\frac{\mathcal{N}(X_i\vert \mu_k^{old},{\sigma_k^{old}}^2) \pi_k^{old}}{\sum_{k'}\mathcal{N}(X_i\vert \mu_{k'}^{old},{\sigma_{k'}^{old}}^2) \pi_{k'}^{old}}.
\end{align} $$

* `M step`, for simplicity, denote $$\gamma_{ik}=p(Z_i=k\vert X_i,\theta^{old})$$.

$$ \begin{align}
\sum_i\mathbb{E}_{Z_i\vert X_i,\theta^{old}} \ln p(X_i,Z_i\vert \theta)&=\sum_i\sum_k p(Z_i=k\vert X_i,\theta^{old}) \ln p(Z_i=k,X_i\vert \theta)\\
&=\sum_k \sum_i p(Z_i=k\vert X_i,\theta^{old}) \ln \mathcal{N}(X_i\vert \mu_k,{\sigma_k}^2) \pi_k\\
&=\sum_k \sum_i \gamma_{ik} \ln \mathcal{N}(X_i\vert \mu_k,{\sigma_k}^2) \pi_k.
\end{align} $$

Take the derivative with respect to $$\mu_k,\sigma_k^2$$, we get

$$ \begin{align}
\mu_k &= \frac{\sum_i X_i \gamma_{ik}}{\sum_{i}\gamma_{ik}}\\
\sigma_k^2 &= \frac{\sum_i (X_i-\mu_k)^2 \gamma_{ik}}{\sum_{i}\gamma_{ik}}.
\end{align} $$

The $$\pi_k$$ is a little more complicated as it is contrained under $$\sum_k \pi_k=1$$. We could either solve it through Lagrangian or make an analogy with the log likelihood of a Multinomial distribution. The latter can be achived by viewing $$\sum_i\gamma_ik$$ as the count for Category $$k$$. Thus,
$$\pi_k=\frac{\sum_{i}\gamma_{ik}}{\sum_{i,k}\gamma_{ik}}.$$

## Variations

### Replacing $$\arg\max$$
In the original formalization, we try to maximize $$L(q,\theta)$$ in each step. However, it usually requires we iterate over all observed data ($$X$$) and sometimes solving $$Q(\theta\vert \theta^{old})$$ may be intractable. The idea here is, instead of calculating the maximum value, we could get similar outcome as long as we increase lower bound $$L(q,\theta)$$. Namely,

* `E step`, fix $$\theta$$ as $$\theta^{old}$$, $$q=\text{increase}_{q}L(q,\theta^{old})$$
* `M step`, fix $$q$$ as $$q^{old}$$, $$\theta=\text{increase}_{\theta}L(q^{old},\theta)$$

The `increase` version of `M step` is called _generalized EM_ which mostly deals with intractable likelihood functions w.r.t. $$\theta$$.

The counterpart of `E step` is called [incremental EM](http://www.cs.toronto.edu/~fritz/absps/emk.pdf). Namely, instead of updating for full $$q(Z)=p(Z\vert X,\theta^{old})$$, we could only update date for a specific data point $$i$$: $$q(Z_i)=p(Z_i\vert X_i,\theta^{old})$$ while keeping the rest unchanged. This update would surely increase $$L(q,\theta)$$ as long as the i.i.d. property holds. Another benefit in this method is the single data point update would induce simpler `M step` update through sufficient statistics (in unprecise words, the new value could be computed by few arithmetics with the old value):
e.g. in the Gaussian mixture model,

$$ \begin{align}
\mu_{k}&=\mu_{k}^{old}+\frac{\gamma_{ik}-\gamma_{ik}^{old}}{N_k}(X_i-\mu_k^{old})\\
N_k&=N_k^{old}+\gamma_{ik}-\gamma_{ik}^{old},
\end{align} $$

where $$N_k$$ initially is defined as $$N_k=\sum_i \gamma_{ik}$$.

### Maximum a Posterior (MAP)
Instead of maximizing $$\ln p(X\vert \theta)$$, we could use the posterior as may be more preferred by Baysians. Namely, we maximize $$\ln p(\theta\vert X)$$. However, note that

$$ \begin{align}
\ln p(\theta\vert X)&=\ln p(X\vert \theta) + \ln p(\theta) - \ln p(X)\\
&=\ln p(X\vert \theta) + \ln p(\theta) + constant\\
&=L(q,\theta)+DL(q\|p) + \ln p(\theta) + constant.
\end{align}$$

The fact that $$L(q,\theta)+\ln p(\theta)$$ is a lower bound for $$\ln p(\theta\vert X)$$ does not change.

Thus, the `E step` of maximizing $$L(q,\theta)$$ w.r.t. $$q$$ would keep the same since $$\ln p(\theta)$$ does not depend on $$q$$. The `M step` will turn into maximizing $$L(q^{old},\theta)+\ln p(\theta)$$. We can write out the whole process:

* `E step`, calculate $$p(Z\vert X,\theta^{old})$$
* `M step`, compute new $$\theta$$ as $$\theta = \arg\max_{\theta} Q(\theta\vert \theta^{old})+\ln p(\theta)$$
