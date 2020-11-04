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

- Improvement over $\tau$
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

## Relative Density Ratio

- The $\gamma$-*relative* density ratio
$$
r\_{\gamma}(\mathbf{x}) = \frac{\ell(\mathbf{x})}{\gamma \ell(\mathbf{x}) + (1 - \gamma) g(\mathbf{x})}
$$
- where $\gamma \ell(\mathbf{x}) + (1 - \gamma) g(\mathbf{x})$ is the $\gamma$-*mixture density* 
  - for some mixing proportion $0 \leq \gamma < 1$

----

## Ordinary Density Ratio

- The *ordinary* density ratio
$$
r\_{0}(\mathbf{x}) = \frac{\ell(\mathbf{x})}{g(\mathbf{x})}
$$

---

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

Vapnik's principle

---

## Class-Probability Estimation (CPE)

---

## Controls

- Next: `Right Arrow` or `Space`
- Previous: `Left Arrow`
- Start: `Home`
- Finish: `End`
- Overview: `Esc`
- Speaker notes: `S`
- Fullscreen: `F`
- Zoom: `Alt + Click`
- [PDF Export](https://github.com/hakimel/reveal.js#pdf-export): `E`

---

## Code Highlighting

Inline code: `variable`

Code block:
```python
porridge = "blueberry"
if porridge == "blueberry":
    print("Eating...")
```

---

## Math

In-line math: $x + y = z$

Block math:

$$
f\left( x \right) = \;\frac{{2\left( {x + 4} \right)\left( {x - 4} \right)}}{{\left( {x + 4} \right)\left( {x + 1} \right)}}
$$

---

## Fragments

Make content appear incrementally

```
{{%/* fragment */%}} One {{%/* /fragment */%}}
{{%/* fragment */%}} **Two** {{%/* /fragment */%}}
{{%/* fragment */%}} Three {{%/* /fragment */%}}
```

Press `Space` to play!

{{% fragment %}} One {{% /fragment %}}
{{% fragment %}} **Two** {{% /fragment %}}
{{% fragment %}} Three {{% /fragment %}}

---

A fragment can accept two optional parameters:

- `class`: use a custom style (requires definition in custom CSS)
- `weight`: sets the order in which a fragment appears

---

## Speaker Notes

Add speaker notes to your presentation

```markdown
{{%/* speaker_note */%}}
- Only the speaker can read these notes
- Press `S` key to view
{{%/* /speaker_note */%}}
```

Press the `S` key to view the speaker notes!

{{< speaker_note >}}
- Only the speaker can read these notes
- Press `S` key to view
{{< /speaker_note >}}

---

## Themes

- black: Black background, white text, blue links (default)
- white: White background, black text, blue links
- league: Gray background, white text, blue links
- beige: Beige background, dark text, brown links
- sky: Blue background, thin dark text, blue links

---

- night: Black background, thick white text, orange links
- serif: Cappuccino background, gray text, brown links
- simple: White background, black text, blue links
- solarized: Cream-colored background, dark green text, blue links

---

{{< slide background-image="/img/boards.jpg" >}}

## Custom Slide

Customize the slide style and background

```markdown
{{</* slide background-image="/img/boards.jpg" */>}}
{{</* slide background-color="#0000FF" */>}}
{{</* slide class="my-style" */>}}
```

---

## Custom CSS Example

Let's make headers navy colored.

Create `assets/css/reveal_custom.css` with:

```css
.reveal section h1,
.reveal section h2,
.reveal section h3 {
  color: navy;
}
```

---

# Questions?

[Ask](https://spectrum.chat/academic)

[Documentation](https://sourcethemes.com/academic/docs/managing-content/#create-slides)

---

## References

- Bergstra, J. S., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-parameter Optimization. In *Advances in Neural Information Processing Systems* (pp. 2546-2554).
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2011). Relative Density-ratio Estimation for Robust Distribution Comparison. In *Advances in Neural Information Processing Systems* (pp. 594-602).