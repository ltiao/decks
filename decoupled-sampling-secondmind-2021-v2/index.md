---
title: Decoupled Sampling | Secondmind
theme: serif
highlightTheme: atom-one-dark
revealOptions:
    transition: convex
    controls: true
    progress: true
---

### Decoupled Sampling from Gaussian Processes

#### Phase I: Quadrature Fourier features 

***

Louis C. Tiao

---

### Sparse GPs
- Let
$$
\begin{bmatrix}
\mathbf{f}\_\ell \newline
\mathbf{f}\_m
\end{bmatrix} \sim
\mathcal{N}\left(  
\begin{bmatrix}
\mathbf{0} \newline
\mathbf{0}
\end{bmatrix},
\begin{bmatrix}
\mathbf{K}\_{\ell \ell} & \mathbf{K}\_{\ell m} \newline
\mathbf{K}\_{m \ell} & \mathbf{K}\_{m m} 
\end{bmatrix}
\right)
$$
- Then
$$
p(\mathbf{f}\_\ell \vert \mathbf{f}\_m = \mathbf{u}) = 
\mathcal{N}(\mathbf{m}, \mathbf{S})
$$
- where $\mathbf{m} = \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} \mathbf{u}$,
- and $\mathbf{S} = \mathbf{K}\_{\ell \ell} - \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} \mathbf{K}\_{m \ell}$.

----

- Matheron's update $(\mathbf{f}\_\ell \vert \mathbf{f}\_m = \mathbf{u}) \stackrel{D}{=} \mathbf{f}\_\ell + \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} (\mathbf{u} - \mathbf{f}\_m)$
- Mean
$$
\begin{align}
\mathbb{E}[\mathbf{f}\_\ell + \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} (\mathbf{u} - \mathbf{f}\_m)] & = 
\underbrace{\mathbb{E}[\mathbf{f}\_\ell]}\_{\mathbf{0}} + \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} (\mathbf{u} - \underbrace{\mathbb{E}[\mathbf{f}\_m]}\_{\mathbf{0}}) \newline & = \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} \mathbf{u} = \mathbf{m}
\end{align}
$$
- Covariance
$$
\begin{align}
\mathrm{Cov}[\mathbf{f}\_\ell + \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} (\mathbf{u} - \mathbf{f}\_m)] & = 
\mathrm{Cov}[\mathbf{f}\_\ell] - \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} \mathrm{Cov}[\mathbf{f}\_m, \mathbf{f}\_\ell] \newline & =
\mathbf{K}\_{\ell \ell} - \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} \mathbf{K}\_{m \ell} = \mathbf{S}
\end{align}
$$

---

### Exact GPs
- Let
$$
\begin{bmatrix}
\mathbf{f}\_\ell \newline
\mathbf{f} + \boldsymbol{\epsilon}
\end{bmatrix} \sim
\mathcal{N}\left(  
\begin{bmatrix}
\mathbf{0} \newline
\mathbf{0}
\end{bmatrix},
\begin{bmatrix}
\mathbf{K}\_{\ell \ell} & \mathbf{K}\_{\ell n} \newline
\mathbf{K}\_{n \ell} & \mathbf{K}\_{n n} +  \sigma^2 \mathbf{I}
\end{bmatrix}
\right)
$$
- Then
$$
p(\mathbf{f}\_\ell \vert \mathbf{f} + \boldsymbol{\epsilon} = \mathbf{y}) = 
\mathcal{N}(\mathbf{m}, \mathbf{S})
$$
- where $\mathbf{m} = \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} \mathbf{y}$,
- and $\mathbf{S} = \mathbf{K}\_{\ell \ell} - \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} \mathbf{K}\_{n \ell}$.

----

- Matheron's update 
$$
(\mathbf{f}\_\ell \vert \mathbf{f} + \boldsymbol{\epsilon} = \mathbf{y}) \stackrel{D}{=} 
\mathbf{f}\_\ell + \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{f} - \boldsymbol{\epsilon})
$$
- Mean
$$
\begin{align}
& \mathbb{E}[\mathbf{f}\_\ell + \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{f} - \boldsymbol{\epsilon})] \newline & = 
\underbrace{\mathbb{E}[\mathbf{f}\_\ell]}\_{\mathbf{0}} + \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} (\mathbf{y} - \underbrace{\mathbb{E}[\mathbf{f} + \boldsymbol{\epsilon}]}\_{\mathbf{0}}) \newline & = \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} \mathbf{y} = \mathbf{m}
\end{align}
$$
- Covariance
$$
\begin{align}
\mathrm{Cov}[\mathbf{f}\_\ell + \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} (\mathbf{u} - \mathbf{f}\_m)] & = 
\mathrm{Cov}[\mathbf{f}\_\ell] - \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} \mathrm{Cov}[\mathbf{f}\_m, \mathbf{f}\_\ell] \newline & =
\mathbf{K}\_{\ell \ell} - \mathbf{K}\_{\ell m} \mathbf{K}\_{m m}^{-1} \mathbf{K}\_{m \ell} = \mathbf{S}
\end{align}
$$

---

### Hybrid update
- Sparse
$$
(\boldsymbol{\Phi}\_\ell \mathbf{w} \vert \boldsymbol{\Phi} \mathbf{w} = \mathbf{u}) \stackrel{D}{=} 
\boldsymbol{\Phi}\_\ell \mathbf{w} + \mathbf{K}\_{\ell m} \mathbf{K}\_{mm}^{-1} (\mathbf{u} - \boldsymbol{\Phi} \mathbf{w})
$$
- Exact
$$
(\boldsymbol{\Phi}\_\ell \mathbf{w} \vert \boldsymbol{\Phi} \mathbf{w} + \boldsymbol{\epsilon} = \mathbf{y}) \stackrel{D}{=} 
\boldsymbol{\Phi}\_\ell \mathbf{w} + \mathbf{K}\_{\ell n} (\mathbf{K}\_{n n} + \sigma^2 \mathbf{I})^{-1} (\mathbf{y} - \boldsymbol{\Phi} \mathbf{w} - \boldsymbol{\epsilon})
$$
