---
title: BORE | NeurIPS2020 Meta-learn Contributed Talk
theme: serif
highlightTheme: monokai
revealOptions:
    transition: none
    controls: true
    progress: true
---

## Bayesian Optimization by Density Ratio Estimation

***

**Louis Tiao**, Aaron Klein, Cédric Archambeau, Edwin Bonilla, Matthias Seeger, and Fabio Ramos

Note:
- Hi. In this talk, we describe an approach to *Bayesian Optimization* by *Density Ratio Estimation*.
- My name is Louis Tiao, and this is a collaboration with Aaron Klein and my colleagues shown here.

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
- By *blackbox*, we usually mean that we only have access to
  - (potentially noisy) observations of the outputs at some given inputs,
  - and useful information such as gradients of the function are generally not available
- then we seek a minimum to this function using as few evaluations as possible 

---

## Bayesian Optimization

- **Probabilistic surrogate model**
  - using past observations $\mathcal{D}\_N = \\{ (\mathbf{x}\_n, y\_n) \\}\_{n=1}^N$
- **Acquisition function** that encodes the *explore-exploit* trade-off
  - derived from **posterior predictive** $p(y | \mathbf{x}, \mathcal{D}\_N)$
  - e.g. *expected improvement (EI)*
<!-- - Output $y \sim \mathcal{N}(f(\mathbf{x}), \sigma^2)$ observed with noise variance $\sigma^2$ -->

Note:
Briefly summarized, BO has *two* components:
- At the core of BO is the *probabilistic surrogate model* of the blackbox function, 
  - build on past observations of input-output pairs *D*,
- Then, BO works by proposing solutions according to an *acquisition function*, 
a function which encodes the trade-off between *exploration* and *exploitation*.
  - usually the acquisition function is built on the properties of surrogate 
  model's **posterior predictive** distribution *p(y|x, D)*.
  - in our talk, we'll be focusing on the widely-popular **expected improvement**, 
  or **EI**, acquisition function.

----

### Utility function: Improvement 

- Improvement over threshold $\tau$
$$
I\_{\gamma}(\mathbf{x}) = \max(\tau - y, 0)
$$
- Note
  - $y$ is a function of $\mathbf{x}$
  - $\gamma$ is a function of $\tau$ (defined next)

Note:
First, let us define the improvement utility function, which quantifies the 
amount of non-negative improvement over some threshold tau.
Note that:
- although *x* doesn't explicitly appear on the RHS, 
it is a function of *y*
- by the same token, *gamma* is function of *tau*, which we define next.

----

### Threshold

- Define threshold $\tau = \Phi^{-1}(\gamma)$ 
  - where $\gamma \in [0, 1)$ is some quantile of observed $y$, i.e.
$$
\gamma = \Phi(\tau) = p(y < \tau)
$$

Note:
- We let *tau* be specified through a function of *gamma*, namely the *inverse CDF* 
of the observed *y* values.
- In simple terms, *tau* is simply some quantile of the *y* values.

----

### Threshold: Examples

1. $\gamma = 0.25$ leads to *first quartile*
2. $\gamma = 0$ leads to $\tau=\min\_n y\_n$ (i.e. conventional defn.)

![Observations](teaser/observations_ecdf_1000x618.png "Observations") <!-- .element height="60%" width="60%" class="plain" -->

Note:
To illustrate we show some example settings of gamma and the thresholds they lead to.
- We're using the same blackbox function from the example in the beginning, along with its observations 
- In the right pane of the figure, we show the empirical CDF of *y* observations.
  - then, we can see that *gamma=0.25* leads to the first quartile of *y* observations, and
  - *gamma=0* leads to the minimum across all *y* observations, which is the 
  conventional setting of the threshold for EI.
- We'll be using this more general definition of the threshold later on.

----

### Expected Improvement (EI)

- Expected value of $I\_{\gamma}(\mathbf{x})$
$$
\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N) = \mathbb{E}\_{\color{red}{p(y | \mathbf{x}, \mathcal{D}\_N)}}[I\_{\gamma}(\mathbf{x})]
$$
  - under posterior predictive $p(y | \mathbf{x}, \mathcal{D}\_N)$
- For **Gaussian** $p(y | \mathbf{x}, \mathcal{D}\_N)$ this leads to a simple 
analytical expression.
- For this reason (and more) **Gaussian processes (GPs)** are the *de facto* 
standard for surrogate models in BO.

Note:
- Finally, we are ready to define the *expected improvement* function,
- which, as the name would suggest, is the *expected value* of the *improvement* 
utility function (just defined), under the **posterior predictive** of the surrogate model. 
- This is why it's important for the **posterior predictive** to be **analytically tractable**.
- Furthermore, it should be noted that, if the predictive is *Gaussian*, this leads to a nice closed-form expression.
- It is for this reason (and many others) that the Gaussian process is arguably the *de facto*
probabilistic model for BO.

----

### Limitations

- **GP-based BO** can also be hampered by the limitations of GPs
  - scalability: $\mathcal{O}(N^3)$ cost [(Titsias, 2009; more)](#)
  - non-stationarity [(Snoek et al, 2014)](#)
  - discrete inputs, ordered or otherwise (categorical) [(Garrido-Merchán and Hernández-Lobato, 2020)](#)
  - conditional dependency structures [(Jenatton et al, 2017)](#)
- **Beyond GPs:** analytical tractability of $\color{red}{p(y | \mathbf{x}, \mathcal{D}\_N)}$ still poses limitations

Note:
- That said, BO based on GPs and also be hampered by their limitations. To name a few,
  - Their *exact* inference scales cubically
  - They assume stationarity
  - Don't deal well with with discrete inputs, or inputs with conditional dependencies.
- As such, to overcome these limitations, much focus has been directed toward extending GPs themselves.
- Further, when considering other models beyond GPs, the need to ensure analytical 
tractability of the predictive poses further limitations on the model's expressiveness.

----

## BO Reimagined

- **Surrogate model is only a means to an end** (i.e. of constructing the acquisition function)
- Alternative formulation?
  - _**bypass** posterior inference altogether?_

Note:
- So, to address these limitiations, 
  - instead of trying to patch the deficiencies of the surrogate model,
  - why don't we step back and re-consider the problem from a different angle.
- First, we must recognize that, at the end of the day, we care about the 
surrogate model insofar as we can use it to construct the acquisition function
- Naturally, we ought to ask: can we formulate the acquisition function in a 
way that doesn't demand analytical tractability on the part of the model.

---

## Density Ratio

The **density ratio** between $\ell(\mathbf{x})$ and $g(\mathbf{x})$
$$
\frac{\ell(\mathbf{x})}{g(\mathbf{x})}
$$

![Ordinary Density Ratio](ordinary/densities_paper_1000x500.png "Ordinary Density Ratio") <!-- .element height="70%" width="70%" class="plain" -->

Note:
Before we do this, let us first introduce some concepts. 
Namely, the *density ratio*:
- let *l* and *g* be a pair of probability distributions.
- Then, the *density ratio* between *l* and *g* is quite simply, the *ratio* of their *densities*.
- This is illustrated in the figure shown here.

----

### Relative Density Ratio

- The $\gamma$-**relative density ratio** between $\ell(\mathbf{x})$ and $g(\mathbf{x})$
$$
r\_{\gamma}(\mathbf{x}) = \frac{\ell(\mathbf{x})}{\gamma \ell(\mathbf{x}) + (1 - \gamma) g(\mathbf{x})}
$$
  where $\gamma \ell(\mathbf{x}) + (1 - \gamma) g(\mathbf{x})$ is the $\gamma$-*mixture density* 
  - for some mixing proportion $0 \leq \gamma < 1$
- For $\gamma = 0$ we recover **ordinary** density ratio $r\_0(\mathbf{x}) = \ell(\mathbf{x}) / g(\mathbf{x})$

Note:
- To generalize this slightly, we introduce the *relative* density ratio, 
which is defined as the ratio between *l* and the mixture density of *l* and *g*
with mixing proportion *gamma*.
- It's easy to see that we recover the *ordinary* density ratio simply by setting *gamma* equal to 0.

----

### Relative Density Ratio Illustrated

![Relative Density Ratio](relative/densities_paper_1000x500.png "Relative Density Ratio") <!-- .element height="90%" width="90%" class="plain" -->

Note:
This is the example from the previous illustration, now also showing the *gamma* relative density ratio with *gamma* set to 0.25.

----

## Ordinary and Relative Density Ratio

- The relative density ratio $r\_{\gamma}(\mathbf{x})$ as a function of the 
ordinary density ratio $r\_0(\mathbf{x})$
$$
r_{\gamma}(\mathbf{x}) = ( \gamma + r_0(\mathbf{x})^{-1} (1 - \gamma) )^{-1}
$$
- Monotonically non-decreasing

Note:
Before moving on, it is worth noting that the relative density ratio *r_gamma* 
can be expressed as a *monotonically non-decreasing* function of the ordinary 
density ratio *r_0*.

---

## BORE: BO by DRE

- Define $\tau$ as before, i.e. $\tau = \Phi^{-1}(\gamma)$
- Then, let $\ell(\mathbf{x})$ and $g(\mathbf{x})$ be distributions s.t.
  - $\mathbf{x} \sim \ell(\mathbf{x})$ if $y < \tau$
  - $\mathbf{x} \sim g(\mathbf{x})$ if $y \geq \tau$

![Observations](teaser/summary_1000x618.png "Observations") <!-- .element height="55%" width="55%" class="plain" -->

Note:
- Now we return to the problem at hand. Namely, finding an alternative 
expression of the EI acquisition function.
- To do this, let's first define *tau* as a function of *gamma* like before, 
with *tau* being the *gamma*-quantile of the *y* values. 
- Then, let us partition the observations, such that 
  - if the output *y* is less than *tau*, then the corresponding input *x* is distributed according to *l*.
  - Otherwise, it is distributed according to *g*.

----

## Define Conditional

- Instead of predictive $p(y | \mathbf{x}, \mathcal{D}\_N)$
  - Specify $p(\mathbf{x} | y, \mathcal{D}\_N)$ in terms of $\ell(\mathbf{x})$ and 
$g(\mathbf{x})$
$$
p(\mathbf{x} | y, \mathcal{D}\_N) = 
\begin{cases} 
  \ell(\mathbf{x}) & \text{if } y < \tau, \newline
  g(\mathbf{x}) & \text{if } y \geq \tau
\end{cases}
$$

Note:
- Now, rather than doing posterior inference to derive the predictive, let us
instead specify the conditional *p(x|y)* directly.
- In particular, if *y* is less than *tau*, we let it be equal to density *l*.
- Otherwise, we let it be equal to *g*.

----

### Relationship: EI and Density Ratio

- [Bergstra et al. 2011](#) demonstrate
$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
$$

Note:
- Then, remarkably, under these definitions, EI is equivalent to the *gamma* relative density ratio, up to a constant factor. 
- As was demonstrated by Bergstra et al. in 2011.

----

## Problem Reformulation

- Reduces maximizing *EI* to maximizing the *relative density ratio*
$$
\begin{align}
\mathbf{x}\^{\star} 
&= \color{red}{\operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}} \newline
&= \color{green}{\operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{r\_{\gamma}(\mathbf{x})}}
\end{align}
$$

Note:
- And so, since EI is proportional to the relative ratio, we can reduce the 
problem of maximizing the former to that of maximizing the latter.
- Thus, effectively allowing us to bypass posterior inference in the surrogate
model.

---

## Tree-structured Parzen Estimator (TPE)

TPE approach [(Bergstra et al. 2011)](#) for maximizing $r\_{\gamma}(\mathbf{x})$
1. Maximize $r\_0(\mathbf{x})$ instead
$$
\begin{align}
\mathbf{x}\^{\star} 
&= \color{red}{\operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{r\_{\gamma}(\mathbf{x})}} \newline
&= \color{green}{\operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{r\_0(\mathbf{x})}}
\end{align}
$$

Note:
- So now we're left with the problem of maximizing *r_gamma*.
- To do this Bergstra et al. propose the TPE method, which makes two choices.
- First, recall from earlier that r_gamma is a **monotonically non-decreasing** function
of *r_0*. 
- Therefore, they simply restrict their attention to maximizing *r_0* instead. 

----

### Shortcomings

- **Singularities.** $r\_0(\mathbf{x})$ is often undefined.
  In contrast, $r\_{\gamma}(\mathbf{x})$ is always well-defined 
  - bounded above by $\gamma^{-1}$ when $\gamma > 0$ [(Yamada et al. 2011)](#)
  - easy to find examples e.g. $\ell(x) = \mathcal{N}(0, 1)$ and $g(x) = \mathcal{N}(0.5, 1)$

![Density Ratio Singularities](singularities/densities_paper_1000x500.png "Density Ratio Singularities") <!-- .element height="50%" width="50%" class="plain" -->

Note:
While this may seem like a nice simplification, it may have unintended 
consequences. In particular, 
- r_0 is unbounded, often diverging to infinity even in simple toy scenarios.
- By contrast r_gamma is always bounded above by *gamma-inverse* for all 
positive *gamma*
- It is easy to find examples of this. In particular, take a zero-mean, 
unit-variance Normal distribution. By shifting the mean by 0.5 and taking the
density ratios, we find that r_0 diverges to infinity while r_gamma remains
upper bounded by 4, which is equivalent to *gamma-inverse* for *gamma = 0.25*.

----

## Tree-structured Parzen Estimator (TPE) II

2. To estimate $r\_0(\mathbf{x})$  
  - separately estimate $\ell(\mathbf{x})$ and $g(\mathbf{x})$ using a 
  tree-based variant of **kernel density estimation (KDE)**
  - take the ratio of the estimators

Note:
The next component of the TPE approach is to estimate r_0, which they do 
simply by 
- individually estimating *l* and *g* using a tree-based variant of kernel 
density estimation, and then
- taking the ratio of these estimators

----

## Advantages

- Computational cost $\mathcal{O}(N)$ instead of $\mathcal{O}(N^3)$ in GP inference
- Equipped to handle tree-structured, continuous, and discrete (ordered and unordered) inputs

Note:
It's clear why this approach may be favorable compared to GP regression.
- We we now incur a linear cost as opposed to the cubic complexity of 
posterior inference in GPs
- Furthermore, it is naturally equipped to deal with tree-structured, continuous, and discrete inputs.
- In spite of these advantages, TPE is not without its shortcomings, as we 
discuss next.

----

## Shortcomings II

- **Vapnik's principle.** "When solving a problem, don't try to solve a more general problem as an intermediate step"
  - *density* estimation is arguably more general and difficult problem than *density ratio* estimation

Note:

In particular, estimating the individual densities is actually a more 
cumbersome approach for a number of reasons. 
Firstly, it violates Vapnik's principle, paraphrased, suggests to us that when 
solving a problem of interest, one should refrain from solving a 
more general problem as an intermediate step.
- And in this instance, *density* estimation is a more general problem; 
  arguably one that is more difficult than *density ratio* estimation.

----

### Shortcomings III

- **Kernel bandwidth.** KDE depends crucially on the selecting
appropriate kernel bandwidths
- **Error sensitivity.** Estimating *two* densities 
  - Optimal bandwidth for estimating a *density* may be detrimental to estimating the *density ratio*
  - Unforgiving to errors in denominator $g(\mathbf{x})$
- **Curse of dimensionality.** KDE often struggles in higher dimensions.
- **Ease of optimization.** Need to maximize ratio of KDEs for candidate suggestion.

Note:
- One of the things that make density estimation so hard is the 
selection of the kernel bandwidth, which is notoriously difficult.
- This is exacerbated by the fact that we need to simultaneously estimate *two* 
densities, 
  - wherein the optimal bandwidth for one of the individual densities 
  may be detrimental to estimating the density ratio as a whole.
  - This factor makes this approach unforgiving to any error in estimating the 
  individual densities, particularly in that of the denominator *g* which has 
  an outsized influence on the resulting density ratio
- For these reasons and more, KDE often falls short in higher dimensions
- And finally, we ultimately care about *optimizing* the density ratio 
in order to suggest candidates. The ratio of KDEs of cumbersome to work with 
in this regard.

----

## Solutions?

- How to avoid the pitfalls of the TPE approach?
  - _**directly** estimate the relative density ratio_

Note:
- Given all these pitfalls we discussed, it stands to reason that we should be
looking for ways to *directly estimate* the relative density ratio.

---

## Density Ratio Estimation 

- KMM: Kernel Mean Matching (Huang et al. 2007) 
<!-- .element: class="fragment fade-out" data-fragment-index="2" -->
- KLIEP: KL Importance Estimation Procedure (Sugiyama et al. 2008) 
<!-- .element: class="fragment fade-out" data-fragment-index="2" -->
- (R)uLSIF : (Relative) Least-squares Important Fitting (Kanamori et al. 2009; Yamada et al. 2011) 
<!-- .element: class="fragment fade-out" data-fragment-index="2" -->
- CPE: Class-Probability Estimation (Qin 1998, Bickel et al. 2007) 
<!-- .element: class="fragment fade-up" data-fragment-index="1" -->

Note:
And how can we go about this? Well, 
- Many density ratio estimation methods methods have already been developed, some of them very sophisticated.
- But for now let's just see how far we can get with a simple baseline.
- Namely, that of *class-probability estimation*.

----

## Class-Probability Estimation (CPE)

- Density ratio estimation is tightly-linked to class-probability estimation
(Qin 1998, Bickel et al. 2007)
- What about **relative** density ratio estimation?

Note:
- It's been long-established that **density ratio estimation** is closely-linked to **class-probability estimation**.
- Therefore, it stands to reason that this applies to the **relative** density ratio as well.

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
- Let $\pi(\mathbf{x})$ abbreviate **class-posterior probability**
$$
\pi(\mathbf{x}) = p(z = 1 | \mathbf{x})
$$

Note:
- To see this, let us define a classification problem by introducing binary 
  labels *z*.
  - which belongs to the positive class, if output *y* is less than *tau*,
  - otherwise, it belongs to the negative class 
- Further, we let *pi* denote the *class-posterior probability*, that is, the 
  *posterior probability* of *x* belonging to the positive class.

----

### Relationship: Density Ratio and Class-Posterior Probability

<!-- 
- ordinary density ratio
$$
r\_0(\mathbf{x}) = \left ( \frac{\gamma}{1 - \gamma} \right )^{-1} \frac{\pi(\mathbf{x})}{1 - \pi(\mathbf{x})}
$$ -->

- The $\gamma$-relative density ratio is equivalent to the class-posterior 
probability up to constant factor
$$
\underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio} = 
\gamma^{-1}
\cdot
\underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$

Note:
Then, it is easy to verify that relative density ratio is exactly equivalent 
to the class-posterior probability, up to constant factor *gamma inverse*.

----

## Quick Recap

$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
\propto \underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$

This is great news! <!-- .element: class="fragment fade-in-then-out" -->

Class-posterior probability $\pi(\mathbf{x})$ can be approximated by training a probabilistic classifier! <!-- .element: class="fragment" -->

Note: 
So, just to quickly recap:
- EI is proportional to the relative density ratio, which is in turn 
proportional to the class-posterior probability.
- This is actually great news, because we can approximate the last of these,
simply by training a probabilistic classifier.

---

### EI by Classifier Training

- We've reduced the problem of computing EI to that of training a classifier
  - Enjoy the benefits of different state-of-the-art classifiers
  - Retain the advantages of TPE and avoid its pitfalls
  - Build arbitrarily expressive classifiers

Note:

So, we've now reduced the problem of computing EI to that of training 
a classifier, which is something we know how to do pretty well,
- this allows us to enjoy the strengths offered by different classifiers,
- all the while, retaining the advantages of TPE while avoiding its pitfalls
- depending on the choice of classifier, we can build arbitrarily expressive 
approximators that have the capacity to deal to with complex, non-linear and 
non-stationary phenomena.

----

### Example: Neural Network Classifier

An obvious choice is a *feed-forward neural network*
- universal approximation
- easily scalable with stochastic optimization
- differentiable end-to-end wrt inputs $\mathbf{x}$

Notes:
An obvious candidate for parameterizing this classifier is a neural network,
- not only for their universal approximation guarantees,
- but also because it is easy to scale up their parameter learning with 
stochastic optimization.
- This is not to mention they are differentiable end-to-end wrt inputs *x*, which
means we can take advantage of methods like L-BFGS for candidate suggestion.

---

## Algorithm

- Classifier $\pi\_{\boldsymbol{\theta}}(\mathbf{x})$ with parameters $\boldsymbol{\theta}$
  1. **Train classifier** $\boldsymbol{\theta}^{*} \gets \operatorname{argmin}\_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$ (loss $\mathcal{L}$) 
  2. **Suggest candidate** $\mathbf{x}\_N \gets \operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}} \pi\_{\boldsymbol{\theta}}(\mathbf{x})$
  3. **Evaluate** $y\_N \gets f(\mathbf{x}\_N)$
  4. **Update dataset** $\mathcal{D}\_N \gets \mathcal{D}\_{N-1} \cup \{ (\mathbf{x}\_N, y\_N) \}$
  5. **Repeat 1.**

Note:
To summarize the algorithm, the so-called BO loop looks like this.
- Let *pi_theta* be the classifier with parameters *theta*
1. Then, we train the classifier by minimizing parameters *theta* wrt an 
appropriate classification loss, such as the log-loss.
2. We suggest a candidate by maximizing input *x* wrt the classifier.
3. Then we evaluate the function at the suggested point, and update the dataset as usual
4. And then we repeat this until some budget is exhausted.

----

## Code

<pre>
  <code data-line-numbers="4|7-14|25-30|32-33|35-36|38-39|41-43">
  import numpy as np

  from bore.models import MaximizableSequential
  from tensorflow.keras.layers import Dense

  # build model
  classifier = MaximizableSequential()
  classifier.add(Dense(16, activation="relu"))
  classifier.add(Dense(16, activation="relu"))
  classifier.add(Dense(1, activation="sigmoid"))

  # compile model
  classifier.compile(optimizer="adam", loss="binary_crossentropy")

  features = []
  targets = []

  # initialize design
  features.extend(features_initial_design)
  targets.extend(targets_initial_design)

  for i in range(num_iterations):

      # construct classification problem
      X = np.vstack(features)
      y = np.hstack(targets)

      tau = np.quantile(y, q=0.25)
      z = np.less(y, tau)

      # update classifier
      classifier.fit(X, z, epochs=200, batch_size=64)

      # suggest new candidate
      x_next = classifier.argmax(bounds=bounds, method="L-BFGS-B", num_start_points=3)

      # evaluate blackbox
      y_next = blackbox.evaluate(x_next)

      # update dataset
      features.append(x_next)
      targets.append(y_next)
  </code>
</pre>

---

# Results

Note:
So, how well does this actually work?

---

## Challenging synthetic test problems

Note:
First, we evaluate our method against TPE on a number of challenging synthetic
test problems for optimization.

----

## Branin (2D)

![Branin](branin/regret_iterations_paper_1000x618.png "Branin") <!-- .element height="80%" width="80%" class="plain" -->

Note:
- Here are the results on the Branin function, a 2D problem.
- The *y-axis* shows the *immediate regret*, defined as the *absolute error* 
between the *global minimum* and the lowest function value observed so far. 
- This is plotted against the *number of function evaluations* on the *x-axis*
which gives us an indication of the methods' sample efficiency.
- BORE is shown in *red*, while TPE is shown in *green*. 
- As we can see, BORE competes favorably against TPE.

----

## Six-hump Camel (2D)

![Six-hump camel](six_hump_camel/regret_iterations_paper_1000x618.png "Six-hump camel") <!-- .element height="80%" width="80%" class="plain" -->

Note:
The story is the same on the **six-hump camel** function, another 2D problem, 
but one with 6 local minima. 

----

## Michalewicz5D (5D)

![Michalewicz 5D](michalewicz_005d/regret_iterations_paper_1000x618.png "Michalewicz 5D") <!-- .element height="80%" width="80%" class="plain" -->

Note:
Here are the results on the 5D Michalewicz function.

----

## Hartmann6D

![Hartmann 6D](hartmann6d/regret_iterations_paper_1000x618.png "Hartmann 6D") <!-- .element height="80%" width="80%" class="plain" -->

Note:
- And finally, on the 6D Hartmann function.
- And indeed, across these problems, we see that BORE consistently outperforms TPE.

---

## Meta-surrogate benchmarks for AutoML

- Klein, A., Dai, Z., Hutter, F., Lawrence, N., & Gonzalez, J. (2019). **Meta-surrogate Benchmarking for Hyperparameter Optimization.**
In *Advances in Neural Information Processing Systems* (pp. 6270-6280).

Note:
We also consider the **meta-surrogate benchmarks for AutoML** by **Klein et al. 2019**

----

## MetaSVM

![MetaSVM](meta_surrogate/ranks_svm.png "MetaSVM") <!-- .element height="80%" width="80%" class="plain" -->

Note:
- And here we show the *average ranks* of each method against function evaluations.
- In addition to TPE, we compare against BO methods with different types of 
surrogate models, specifically, BO based on GPs, SMAC based on random forests, 
and of course, TPE.
- We also compare against SOTA evolutionary algorithms which are often 
competitive against BO methods.
- Across different problem classes, BORE (shown in blue) either outperforms all 
other methods or achieves comparable performance
- We see this for the SVM problem class

----

## MetaFCNet

![MetaFCNet](meta_surrogate/ranks_fcnet.png "MetaFCNet") <!-- .element height="80%" width="80%" class="plain" -->

Note:
- The FCNet problem class

----

## MetaXGBoost

![MetaXGBoost](meta_surrogate/ranks_xgboost.png "MetaXGBoost") <!-- .element height="80%" width="80%" class="plain" -->

Note:
- And the XGBoost problem class

---

## Final Recap

1. Problem of computing EI can be reduced to that of probabilistic classification
$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
\propto \underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$
2. The TPE method falls short in important ways
3. Simple implementation based on feed-forward NN delivers promising results

Note:
- In this talk, we discussed how the problem of *computing EI* can be reduced to 
that of *probabilistic classification*. 
- This observation is made through 
  * the well-known link between **probabilistic classification** and **density ratio estimation**, and 
  * the lesser-known insight that EI 
can be expressed as the **relative density ratio** between two unknown 
distributions.
- We dissected the TPE method: an approach to exploit the latter link, and discussed its failure modes.
- Finally, we proposed an alternative that avoids these failure modes, and showed that a simple implementation of this competes favorably with SOTA methods.

---

## Conclusion

- **Simplicity** and **effectiveness** makes BORE an attractive approach
- **Extensibility** of the BORE framework offers numerous exciting avenues for future exploration 

Note:
Overall, we conclude that
- BORE is a simple but effective framework for BO
- Its high degree of extensibility offers many promising future directions

---

## Thanks for watching!

Note:
- Thank you very much for watching
- I am looking forward to having discussions with you :)

---

## References

- Bergstra, J. S., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-parameter Optimization. In *Advances in Neural Information Processing Systems* (pp. 2546-2554).
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2011). Relative Density-ratio Estimation for Robust Distribution Comparison. In *Advances in Neural Information Processing Systems* (pp. 594-602).

----

## References II
