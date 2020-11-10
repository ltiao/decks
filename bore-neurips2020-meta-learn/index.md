---
title: BORE | NeurIPS2020 Meta-learn Contributed Talk
theme: serif
highlightTheme: monokai
revealOptions:
    transition: slide
    controls: true
    progress: true
---

## Bayesian Optimization by Density Ratio Estimation

***

**Louis Tiao**, Aaron Klein, Cédric Archambeau, Edwin Bonilla, Matthias Seeger, and Fabio Ramos

Note:
- Hi, I am Louis Tiao, and in this talk, we will discuss *BORE*, an approach to *Bayesian Optimization through Density Ratio Estimation*.
- This is joint work done with Aaron Klein, and other collaborators listed here.

---

## Blackbox optimization

Find input $\mathbf{x}$ that minimizes blackbox function $f(\mathbf{x})$
$$
\mathbf{x}^{\star} = \operatorname{argmin}\_{\mathbf{x} \in \mathcal{X}}{f(\mathbf{x})}
$$

![Observations](teaser/observations_1000x618.png "Observations") <!-- .element height="60%" width="60%" class="plain" -->

Note:
First of all, some background: Bayesian optimization is one of the most effective and
widely-used methods for the *global* optimization of *blackbox* functions. 
- By *blackbox*, we usually mean that we observe *no other* information about 
  the function, other than its outputs given some inputs *x*,
  - in particular, gradients of the function are not available
- In the figure shown on this slide, we see
  - noisy observations of the outputs plotted against their corresponding inputs, 
  - and these are shown against the backdrop of the latent blackbox function 
  which we cannot directly observe, but nonetheless wish to minimize

---

## Bayesian Optimization

- **Probabilistic surrogate model**
  - using past observations $\mathcal{D}\_N = \\{ (\mathbf{x}\_n, y\_n) \\}\_{n=1}^N$
- **Acquisition function** that encodes the explore-exploit trade-off
  - derived from posterior predictive distribution $p(y | \mathbf{x}, \mathcal{D}\_N)$
  - e.g. *expected improvement (EI)*
<!-- - Output $y \sim \mathcal{N}(f(\mathbf{x}), \sigma^2)$ observed with noise variance $\sigma^2$ -->

Note:
Briefly summarized, BO has *two* components:
- At the core of BO is the *probabilistic surrogate model*, a model of the function outputs, 
  - which is learned from past observations of input-output pairs *D*,
  - and that can provide uncertainty estimates over outputs
- Then, BO works by proposing solutions according to an *acquisition function*, 
a function which encodes the trade-off between exploration and exploitation.
  - usually a function of posterior predictive of the surrogate model
  - in our talk, we'll be focussing on the widely-popular **expected improvement**, 
    or **EI**, acquisition function.

----

### Utility function: Improvement 

- Improvement over threshold $\tau$
$$
I\_{\gamma}(\mathbf{x}) = \max(\tau - y, 0)
$$
- Note
  - $y$ is a function of $\mathbf{x}$
  - $\tau$ is a function of $\gamma$ (defined next)

Note:
- non-negative improvement over tau

----

### Threshold

- Define threshold $\tau = \Phi^{-1}(\gamma)$ 
  - where $\gamma$ is some quantile of observed $y$, i.e.
$$
\gamma = \Phi(\tau) = p(y < \tau)
$$
  - for example, $\gamma=0$ leads to $\tau=\min\_n y\_n$

----

### Threshold: Examples

1. $\gamma = 0.25$
2. $\gamma = 0$ leading to $\tau=\min\_n y\_n$

![Observations](teaser/observations_ecdf_1000x618.png "Observations") <!-- .element height="60%" width="60%" class="plain" -->

----

## Expected Improvement (EI)

- Expected value of $I\_{\gamma}(\mathbf{x})$
$$
\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N) = \mathbb{E}\_{\color{red}{p(y | \mathbf{x}, \mathcal{D}\_N)}}[I\_{\gamma}(\mathbf{x})]
$$
  - under posterior predictive $p(y | \mathbf{x}, \mathcal{D}\_N)$
- For Gaussian $p(y | \mathbf{x}, \mathcal{D}\_N)$, this leads to a simple 
analytical expression, **but also imposes constraints**

Note:
- Finally, we are ready to define the *expected improvement*, or *EI* 
acquisition function,
- and, as the name suggests, it's the *expected value* of the *improvement* 
utility function, under the *posterior predictive* of the surrogate model. 
- This reveals the requirement of analytical tractability of the posterior.

----

## Limitations

- Analytical tractability of $\color{red}{p(y | \mathbf{x}, \mathcal{D}\_N)}$ 
  poses limitations
  - scalability
  - stationarity and homeoscedasticity
  - discrete variables, ordered or otherwise (categorical)
  - conditional dependency structures

----

## BO Reimagined

- *The surrogate model is only a means to an end*
  - i.e. constructing the acquisition function
- Alternative formulation?
  - bypass posterior inference altogether?

Note:
- To address these limitiations, 
  - rather than trying to patch the deficiencies of the surrogate model
  - let us step back and re-consider the problem from a different angle
- First, we must recognize that, at the end of the day, we care about the 
surrogate model insofar as we can use it to construct the acquisition function
- Can we formulate the acquisition function in such a way as to bypass 
posterior inference in the surrogate model altogether?

---

## Density Ratio

The density ratio between $\ell(\mathbf{x})$ and $g(\mathbf{x})$
$$
\frac{\ell(\mathbf{x})}{g(\mathbf{x})}
$$

*1d synthetic examples here*

Note:
To do this, let us first introduce the *density ratio*. Namely,
- let *l(x)* and *g(x)* be a pair of probability distributions.
- Then, the *density ratio* between *l(x)* and *g(x)* is simply the ratio of 
their densities.

----

## Relative Density Ratio

- The $\gamma$-*relative* density ratio between $\ell(\mathbf{x})$ and $g(\mathbf{x})$
$$
r\_{\gamma}(\mathbf{x}) = \frac{\ell(\mathbf{x})}{\gamma \ell(\mathbf{x}) + (1 - \gamma) g(\mathbf{x})}
$$
  where $\gamma \ell(\mathbf{x}) + (1 - \gamma) g(\mathbf{x})$ is the $\gamma$-*mixture density* 
  - for some mixing proportion $0 \leq \gamma < 1$
- For $\gamma = 0$ we recover *ordinary* density ratio
$$
r\_0(\mathbf{x}) = \frac{\ell(\mathbf{x})}{g(\mathbf{x})}
$$

Note:
Let us consider a slight generalization

----

## Ordinary and Relative Density Ratio

- The relative density ratio $r\_{\gamma}(\mathbf{x})$ as a function of the 
ordinary density ratio $r\_0(\mathbf{x})$
$$
r_{\gamma}(\mathbf{x}) = ( \gamma + r_0(\mathbf{x})^{-1} (1 - \gamma) )^{-1}
$$
- Monotonically non-decreasing

Note:
Before moving on, please keep in mind that the relative density ratio *r-gamma* 
can be expressed as a *monotonically non-decreasing* function of the ordinary 
density ratio *r-zero*.

---

## BORE: BO by DRE

- Let $\ell(\mathbf{x})$ and $g(\mathbf{x})$ be distributions such that
  - $\mathbf{x} \sim \ell(\mathbf{x})$ if $y < \tau$
  - $\mathbf{x} \sim g(\mathbf{x})$ if $y \geq \tau$

![Observations](teaser/summary_1000x618.png "Observations") <!-- .element height="60%" width="60%" class="plain" -->

Note:
- Now we get to the crux of our work
- In other words, we assume that *x* is distributed according to *l(x)* if its 
corresponding target metric *y < tau*, otherwise, it is distributed according 
to *g(x)*

----

## Define Conditional

- Instead of $p(y | \mathbf{x}, \mathcal{D}\_N)$
- Specify $p(\mathbf{x} | y, \mathcal{D}\_N)$ in terms of $\ell(\mathbf{x})$ and 
$g(\mathbf{x})$
$$
p(\mathbf{x} | y, \mathcal{D}\_N) = 
\begin{cases} 
  \ell(\mathbf{x}) & \text{if } y < \tau, \newline
  g(\mathbf{x}) & \text{if } y \geq \tau
\end{cases}
$$

----

### Relationship: EI and Density Ratio

- [Bergstra et al. 2011](#) demonstrate
$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
$$

Note:
- Under this construction, Bergstra et al. in 2011 showed that EI is equivalent
to the *gamma*-relative density ratio, up to a constant factor. 

----

## Problem Reformulation

- Reduce maximizing EI to maximizing the relative density ratio
$$
\begin{align}
\mathbf{x}\^{\star} 
&= \color{red}{\operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}} \newline
&= \color{green}{\operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{r\_{\gamma}(\mathbf{x})}}
\end{align}
$$

Note:
- Since EI is proportional to the relative ratio, we can reduce the problem of 
maximizing EI to that of maximizing the relative ratio.
- Thus allowing us to bypass posterior inference

---

## Tree-structured Parzen Estimator (TPE)

TPE approach [(Bergstra et al. 2011)](#) for maximizing $r\_{\gamma}(\mathbf{x})$
1. Ignore $\gamma$
$$
\begin{align}
\mathbf{x}\^{\star} 
&= \operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{\color{red}{r\_{\gamma}(\mathbf{x})}} \newline
&= \operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{\color{green}{r\_0(\mathbf{x})}}
\end{align}
$$

----

## Shortcomings

- **Singularities.** $r\_0(\mathbf{x})$ is often undefined.
  In contrast, $r\_{\gamma}(\mathbf{x})$ is always well-defined 
  - bounded above by $\gamma^{-1}$ when $\gamma > 0$ [(Yamada et al. 2011)](#)

----

## Tree-structured Parzen Estimator (TPE) II

2. Tree-based variant of kernel density estimation (KDE)
  - separately estimate $\ell(\mathbf{x})$ and $g(\mathbf{x})$
  - estimate $r\_0(\mathbf{x})$ using the ratio of these estimates  

----

## Shortcomings II

- **Vapnik's principle.** "When solving a problem, don't try to solve a more general problem as an intermediate step"
  - *density* estimation is arguably more general and difficult problem than *density ratio* estimation

Note:

Vapnik's principle, paraphrased, suggests to us that when solving a problem of 
interest, one should refrain from resorting to solve a more general problem as 
an intermediate step.
- And in this instance, *density* estimation is a more general problem that is 
arguably more difficult than *density ratio* estimation.

----

## Shortcomings III

- **Kernel bandwidth.**
- **Error sensitivity.**
- **Curse of dimensionality.**
- **Ease of optimization.**

----

# Solutions?

- How to avoid the pitfalls of the TPE approach?
  - Directly estimate the relative density ratio?

---

## Density Ratio Estimation 

- **CPE: Class-Probability Estimation** (Qin 1998, Bickel et al. 2007)
- KMM: Kernel Mean Matching (Huang et al. 2007)
- KLIEP: KL Importance Estimation Procedure (Sugiyama et al. 2008)
- (R)uLSIF : (Relative) Least-squares Important Fitting (Kanamori et al. 2009; Yamada et al. 2011)

Note:
There's a wealth of knowledge that has been built up on the subject of 
density ratio estimation,

----

## Class-Probability Estimation (CPE)

Density ratio estimation is tightly-linked to class-probability estimation
(Qin 1998, Bickel et al. 2007)

----

## Classification Problem

- Introduce binary label $z$
$$
z =
\begin{cases} 
  1 & \text{if } y < \tau, \newline
  0 & \text{if } y \geq \tau
\end{cases}
$$
- Let $\pi(\mathbf{x})$ abbreviate class-posterior probability 
$$
\pi(\mathbf{x}) = p(z = 1 | \mathbf{x})
$$

----

### Relationship: Density Ratio and Class-Posterior Probability

<!-- 
- ordinary density ratio
$$
r\_0(\mathbf{x}) = \left ( \frac{\gamma}{1 - \gamma} \right )^{-1} \frac{\pi(\mathbf{x})}{1 - \pi(\mathbf{x})}
$$ -->

- The $\gamma$-relative density ratio is exactly equivalent to the 
class-posterior probability, up to constant factor $\gamma^{-1}$
$$
\underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio} = 
\gamma^{-1}
\cdot
\underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$

Note:
- It is well-known that the ordinary density ratio estimation is closely-linked 
to class-probability estimation.
- Therefore, it stands to reason that the relative density ratio is related in 
some way as well.
- It turns out, it is easy to verify that relative density ratio is exactly 
equivalent to the class-posterior probability, up to constant factor 
one-over-*gamma*

----

## Quick Recap

$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
\propto \underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$

---

- This is good news!
  - approximated by probabilistic classifier

- Parameterized function
- Proper scoring rule

---

## BO Loop

Code

---

- reduced the problem of computing EI to that of training a probabilistic classifier
- enjoy the strengths and benefits different state-of-the-art classifiers have to offer
- e.g. feed-forward neural networks:
  - universal approximators
  - easily scalable with stochastic optimization
  - differentiable end-to-end wrt inputs $\mathbf{x}$

Notes:
- last but not least, differentiable end-to-end wrt inputs x

---

# Results

---

## Challenging synthetic test problems

----

## Branin (2D)

![Branin](branin/regret_iterations_paper_1000x618.png "Branin")

----

## Six-hump Camel (2D)

![Six-hump camel](six_hump_camel/regret_iterations_paper_1000x618.png "Six-hump camel")

----

## Michalewicz5D (5D)

![Michalewicz 5D](michalewicz_005d/regret_iterations_paper_1000x618.png "Michalewicz 5D")

----

## Hartmann6D

![Hartmann 6D](hartmann6d/regret_iterations_paper_1000x618.png "Hartmann 6D")

---

## Meta-surrogate benchmarks for AutoML

----

## MetaSVM

----

## MetaFCNet

----

## MetaXGBoost

---

## Final Recap

- Problem of computing EI can be reduced to that of probabilistic classification
$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
\propto \underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$
- TPE method falls short in important ways
- Simple implementation based on feed-forward NN delivers promising results

---

## Conclusion

- **Simplicity** and **effectiveness** makes BORE a promising approach
- **Extensibility** offers many exciting avenues for further exploration 

Note:
- BORE is a simple but effective alternative to conventional BO
- Its extensibility offers 

---

# Questions?

---

## References

- Bergstra, J. S., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-parameter Optimization. In *Advances in Neural Information Processing Systems* (pp. 2546-2554).
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2011). Relative Density-ratio Estimation for Robust Distribution Comparison. In *Advances in Neural Information Processing Systems* (pp. 594-602).

----

## References II
