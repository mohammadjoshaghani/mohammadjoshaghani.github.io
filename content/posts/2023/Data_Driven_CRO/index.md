---
title: Summary of the research paper: "Data-Driven Conditional Robust Optimization"
date: 2023-05-17T00:00:00.000+00:00
description: In this blog I talk about the paper "Data-Driven Conditional Robust Optimization". 
tags: [academic]
---

# Introduction:
## What is CRO and why is it important?

In the contextual optimization problem, a decision maker tries to solve a decision-making problem that has uncertainty in the distribution of parameters of objective or (and) constraints, but some covariates (or features) contribute to the random parameters that can be exploited. 

Having covariates available to aid in determining the distribution of uncertain parameters has been shown to be valuable in many fields. For instance, in scenarios such as navigating through a city, data on weather and time of day can help resolve uncertainties surrounding traffic congestion and assist in finding the shortest route. Similarly, when it comes to portfolio optimization, historical stock prices and market sentiment expressed on Twitter can impact future stock returns.

## Notations:
In a simple cost minimization problem, let $
\mathcal{X} \subseteq \mathbb{R}^n$ and $c(x, \xi)
$ respectively be the feasible
set of actions and a cost that depends on both the action $x$ and a random perturbation vector $\xi \in \mathbb{R}^m$. The decision maker has access to a vector of covariates $\psi \in \mathbb{R}^m$ that are assumed to be correlated to $\xi$.

In the conditional stochastic optimization problem, decision maker tries to identify optimal policy $\boldsymbol{x}^*$,  dependent on $\psi$, that is the solution to the following minimization problem:

$
\begin{equation}
\boldsymbol{x}^*(\psi) \in \underset{x \in \mathcal{X}}{\operatorname{argmin}} \mathbb{E}[c(x, \xi) \mid \psi]
\end{equation}
$

That is, optimal action (policy) is the one in  $\mathcal{X}$ that minimizes the expectation of cost function $c$ when $\psi$ is known.

In the conditional robust optimization problem, decision maker tries to get optimal policy $\boldsymbol{x}^*$, that minimizes worst-case loss, i.e., $\max _{\xi \in \mathcal{U}(\psi)} c(x, \xi)$:

$
\begin{equation}
\boldsymbol{x}^*(\psi):=\underset{x \in \mathcal{X}}{\operatorname{argmin}} \max _{\xi \in \mathcal{U}(\psi)} c(x, \xi),
\end{equation}
$

where, $\xi$ is sampled from ambiguity set $\mathcal{U}(\psi)$, given covariates $\psi$.

# How did related works solve problems related to the CRO?

Previous works exert one-class classification technique in designing uncertainty set for robust optimization. Natarajan et al., 2008 used variance and covariance of historical data. Goerigk and Kurtz, 2020 applied deep neural networks to learn uncertainty set. Fard et al., 2020 tried to jointly learn the lower dimensional representation of data and cluster with k-means. But none of them tried to exploit deep neural networks representation learning to create lower representation of data, and jointly used clustering to create uncertainty set from covariate information correlated to random parameters $\xi$.

# The Deep Data-Driven Robust Optimization (DDDRO) Approach: 
Goerigk and Kurtz, 2020 proposed using deep neural networks to create uncertainty set $\mathcal{U}$ as follow:
$
\begin{equation}
\mathcal{U}(W, R):=\left\{\xi \in \mathbb{R}^m:\left\|f_W(\xi)-\bar{f}_0\right\| \leq R\right\}
\end{equation}
$
where, $f_W: \mathbb{R}^m \rightarrow \mathbb{R}^d$ is a deep neural networks, with parameters $W$, that map random variables $\xi$ to a new representation in which uncertainty set $\mathcal{U}$ can be obtained around center $\bar{f}_0$ with radius $R$.

$\bar{f}_0$ is the center of cluster, where $W_o$ represents some initial random parameters of deep neural networks:

$
\begin{equation}
\bar{f}_0:=(1 / N) \sum_{i \in[N]} f_{W_0}\left(\xi_i\right)
\end{equation}
$

Given dataset $\mathcal{D}_{\xi}=\left\{\xi_1, \xi_2 \ldots \xi_N\right\}$ they suggest to minimize the corresponding loss function of one-class classification, i.e., the empirical centered total variation of the projected data points:

$
\begin{equation}
\min _W \frac{1}{N} \sum_{i=1}^N\left\|f_W\left(\xi_i\right)-\bar{f}_0\right\|^2
\end{equation}
$
The design of the neural network's layers is based on a concept called constraint generation. Constraint generation is about adding filters that make a reduced uncertainty set $\mathcal{U'}$, containing worst-case loss in original set $\mathcal{U}$. Their proposed deep neural network makes uncertainty set $\mathcal{U}(W,R)$: 

$
\begin{equation}
\mathcal{U}(W, R)=\left\{\begin{array}{c}
\begin{array}{c}
\exists u \in\{0,1\}^{d \times K \times L}, \zeta \in \mathbb{R}^{d \times L}, \phi \in \mathbb{R}^{d \times L} \\
\sum_{k=1}^K u_j^{k, \ell}=1, \forall j, \ell \\
\phi^1=W^1 \xi \\
\zeta^L_j = \sum_{j=1}^K u_{j}^{k, \ell} a_k^{\ell} \phi_j^{\ell}+\sum_{k=1}^K u_j^{k, \ell} b_k^{\ell}, \forall j, \ell \\
\phi^{\ell}=W^{\ell} \zeta^{\ell-1}, \forall \ell \geq 2 \\
\sum_{k=1}^K u_j^{k, \ell}{\underline{\alpha}}_k^{\ell} \leq \phi_j^{\ell} \leq \sum_{k=1}^K u_j^{k, \ell} \bar{\alpha}_k^{\ell}, \forall j, \ell \\
\left\|\zeta^L-\bar{f}_0\right\| \leq R
\end{array}
\end{array}\right\},
\end{equation}
$
where $\phi^{\ell}$ is the output at $\ell$-th layer of neural networks, and $a_k^{\ell}, b_k^{\ell}, {\underline{\alpha}}_k^{\ell}, \bar{\alpha}_k^{\ell}$ are parameters that create $K$ affine pieces.

# Deep Data-driven **Conditional** Robust Optimization:

## The Deep “Cluster then Classify” (DCC) Approach:

## The Integrated Deep Cluster-Classify (IDCC) Approach:

## (Can we use Risk-averse instead of risk-neutral measure in this approach?) [Connections to Contextual Value-at-Risk Optimization:]

# Learning Conditional uncertainty set of simulated data:

# Suggestion for future works:

# References: 
Karthik Natarajan, Dessislava Pachamanova, and Melvyn Sim. Incorporating asymmetric distributional information in robust value-at-risk optimization. Management Science, 54(3):573–585,
2008.

Marc Goerigk and Jannis Kurtz. Data-driven robust optimization using unsupervised deep learning.
arXiv preprint arXiv:2011.09769, 2020.

Maziar Moradi Fard, Thibaut Thonet, and Eric Gaussier. Deep k-means: Jointly clustering with
k-means and learning representations. Pattern Recognition Letters, 138:185–192, 2020.