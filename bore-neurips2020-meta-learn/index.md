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

**Louis C. Tiao**, Aaron Klein, Cédric Archambeau, Edwin V. Bonilla, Matthias Seeger, and Fabio Ramos

---

## Blackbox optimization

- Input $\mathbf{x}$ the blackbox function $f : \mathcal{X} \to \mathbb{R}$
- Output $y \sim \mathcal{N}(f(\mathbf{x}), \sigma^2)$ observed with noise variance $\sigma^2$

*Show figure here*

Note:
- blackbox optimization
  - derivative-free optimization
  - global optimization
- input vector
- output scalar

---

## Bayesian Optimization

<!-- Observations $\mathcal{D}\_N = \\{ (\mathbf{x}\_n, y\_n) \\}\_{n=1}^N$ -->

- Surrogate probabilistic model
- Acquisition function to balance explore-exploit

----

### Utility function: Improvement 

- Improvement over threshold $\tau$
$$
I\_{\gamma}(\mathbf{x}) = \max(\tau - y, 0)
$$
- Define threshold $\tau = \Phi^{-1}(\gamma)$ 
  - where $\gamma$ is some quantile of observed $y$, i.e.
$$
\gamma = \Phi(\tau) = p(y < \tau)
$$
  - for example, $\gamma=0$ leads to $\tau=\min\_n y\_n$.

Note:
- non-negative improvement over tau

----

Define $\tau$ and $\gamma$

*Show augmented figure here*

----

## Expected Improvement (EI)

- Posterior predictive
$$
p(y | \mathbf{x}, \mathcal{D}\_N)
$$
- Expected value of $I\_{\gamma}(\mathbf{x})$ under posterior predictive 
$$
\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N) = \mathbb{E}\_{p(y | \mathbf{x}, \mathcal{D}\_N)}[I\_{\gamma}(\mathbf{x})]
$$

----

Analytical tractability poses limitations
- scalability
- stationarity and homeoscedasticity
- discrete variables, ordered or otherwise (categorical)
- conditional dependency structures

---

## Density Ratio

$$
\frac{\ell(\mathbf{x})}{g(\mathbf{x})}
$$

----

## Relative Density Ratio

- The $\gamma$-*relative* density ratio between $\ell(\mathbf{x})$ and $g(\mathbf{x})$:
$$
r\_{\gamma}(\mathbf{x}) = \frac{\ell(\mathbf{x})}{\gamma \ell(\mathbf{x}) + (1 - \gamma) g(\mathbf{x})}
$$
  where $\gamma \ell(\mathbf{x}) + (1 - \gamma) g(\mathbf{x})$ is the $\gamma$-*mixture density* 
  - for some mixing proportion $0 \leq \gamma < 1$
- For $\gamma = 0$ we get the *ordinary* density ratio
$$
r\_0(\mathbf{x}) = \frac{\ell(\mathbf{x})}{g(\mathbf{x})}
$$

----

- The relative density ratio $r\_{\gamma}$ as a function of the ordinary
density ratio $r\_0$
$$
r_{\gamma}(\mathbf{x}) = ( \gamma + r_0(\mathbf{x})^{-1} (1 - \gamma) )^{-1}
$$
- Monotonically non-decreasing

---

- Let $\ell(\mathbf{x})$ and $g(\mathbf{x})$ be distributions such that
  - $\mathbf{x} \sim \ell(\mathbf{x})$ if $y < \tau$
  - $\mathbf{x} \sim g(\mathbf{x})$ if $y \geq \tau$

----

## Conditional

Express $p(\mathbf{x} | y, \mathcal{D}\_N)$ in terms of $\ell(\mathbf{x})$ and 
$g(\mathbf{x})$
$$
p(\mathbf{x} | y, \mathcal{D}\_N) = 
\begin{cases} 
  \ell(\mathbf{x}) & \text{if } y < \tau, \newline
  g(\mathbf{x}) & \text{if } y \geq \tau
\end{cases}
$$

---

## Bergstra et al. 2011

[Bergstra et al. 2011](#/10)

$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
$$

----

- Reduce maximizing EI to maximizing the relative density ratio:

$$
\begin{align}
\mathbf{x}\^{\star} 
&= \color{red}{\operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}} \newline
&= \color{green}{\operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{r\_{\gamma}(\mathbf{x})}}
\end{align}
$$

---

### Tree-structured Parzen Estimator (TPE)

Ignore $\gamma$

$$
\begin{align}
\mathbf{x}\^{\star} 
&= \operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{r\_0(\mathbf{x})} \newline
&= \operatorname{argmax}\_{\mathbf{x} \in \mathcal{X}}{r\_{\gamma}(\mathbf{x})}
\end{align}
$$

***

*Singularities.* 

----

Kernel density estimation

***

- *Vapnik's principle*
- *Kernel bandwidth*
- *Error sensitivity*
- *Curse of dimensionality*
- *Ease of optimization*

---

### Class-Probability Estimation (CPE)

- Density ratio estimation is closely related to class-probability estimation
- Binary variables
$$
z =
\begin{cases} 
  1 & \text{if } y < \tau, \newline
  0 & \text{if } y \geq \tau.
\end{cases}
$$
- class-posterior probability
$$
\pi(\mathbf{x}) = p(z = 1 | \mathbf{x})
$$

----

<!-- 
- ordinary density ratio
$$
r\_0(\mathbf{x}) = \left ( \frac{\gamma}{1 - \gamma} \right )^{-1} \frac{\pi(\mathbf{x})}{1 - \pi(\mathbf{x})}
$$ -->

the $\gamma$-relative density ratio is exactly equivalent to the 
class-posterior probability up to constant factor $\gamma^{-1}$

- relative density ratio
$$
\underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio} = 
\gamma^{-1}
\cdot
\underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$

----

## Quick Recap

$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
\propto \underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$

approximated by probabilistic classifier

---

- Parameterized function
- Proper scoring rule

---

## BO Loop

---

- reduced the problem of computing \gls{EI} to that of training a probabilistic classifier
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

## Branin

----

## Six-hump camel

----

## Michalewicz5D


----

## Hartmann6D

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

- Problem of computing EI can be reduced to that of probabilistic classification:
$$
\underbrace{\alpha\_{\gamma}(\mathbf{x}; \mathcal{D}\_N)}\_\text{expected improvement} \propto \underbrace{r\_{\gamma}(\mathbf{x})}\_\text{relative density ratio}
\propto \underbrace{\pi(\mathbf{x})}\_\text{class-posterior probability} 
$$
- TPE Falls short
- Simple implementation based on feed-forward neural network delivers promising results

---

## Conclusion

- Simplicity and effectiveness make BORE a promising approach
- Extensibility offers many exciting avenues for further exploration 

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
