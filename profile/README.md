# CortexQuant: Compressing High-Dimensional Vectors — Provably Near-Optimal

*A data-oblivious algorithm that achieves near-optimal distortion rates for both MSE and inner product estimation, with zero calibration overhead.*

---

**What if you could compress your KV cache by 5× — and prove, mathematically, that you're within a constant factor of the best possible?**

That's the core premise of CortexQuant, a vector quantization algorithm built from first principles. The result: a quantizer that is online (no calibration, no codebook training), GPU-friendly, and provably near-optimal in both mean-squared error and inner product distortion — for any bit-width and dimension.

---

## Why vector quantization is hard

Vector quantization (VQ) maps a floating-point vector $\mathbf{x} \in \mathbb{R}^d$ to a short binary string, then reconstructs an approximation. The distortion is unavoidable — the question is how small you can make it.

Shannon's source coding theory tells us there's a fundamental lower bound on achievable distortion for any compression scheme. Most practical algorithms — including product quantization (PQ), the industry workhorse — fall far short of this bound. They also require expensive offline training (k-means codebooks), making them unsuitable for online settings like KV cache quantization, where vectors arrive in real time.

> Existing algorithms either have suboptimal distortion bounds relative to bit-width, or they're too slow for online inference. CortexQuant addresses both simultaneously.

We set two distortion objectives. Given a quantizer $Q$ and reconstruction $Q^{-1}$:

$$D_{\tt mse} := \mathbb{E}_{Q}\!\left[\left\| \mathbf{x} - Q^{-1}(Q(\mathbf{x})) \right\|_2^2 \right]$$

$$D_{\tt prod} := \mathbb{E}_{Q}\!\left[\left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, Q^{-1}(Q(\mathbf{x})) \rangle \right|^2 \right]$$

For inner product estimation we additionally require the estimator to be *unbiased*:

$$\mathbb{E}\!\left[ \langle \mathbf{y}, Q^{-1}(Q(\mathbf{x})) \rangle \right] = \langle \mathbf{y}, \mathbf{x} \rangle$$

---

## The key insight: random rotation induces a known distribution

CortexQuant begins by multiplying any input vector $\mathbf{x} \in \mathbb{S}^{d-1}$ by a random rotation matrix $\boldsymbol{\Pi}$. The rotated vector $\boldsymbol{\Pi} \cdot \mathbf{x}$ is now uniformly distributed on the unit hypersphere — regardless of what $\mathbf{x}$ was.

A classical result tells us exactly what each coordinate of such a vector looks like:

**Lemma 1.** For $\mathbf{x}$ uniform on $\mathbb{S}^{d-1}$, each coordinate follows

$$\mathbf{x}_j \sim f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)}\left(1 - x^2\right)^{(d-3)/2}$$

In high dimensions this converges to $\mathcal{N}(0, 1/d)$.

This is powerful for two reasons. First, each coordinate follows a *known* distribution, so we can precompute the optimal scalar quantizer for it exactly. Second, in high dimensions distinct coordinates become nearly independent, so we can quantize them separately without losing optimality.

---

## MSE-optimal CortexQuant

The algorithm is straightforward:

1. Precompute a random rotation matrix $\boldsymbol{\Pi}$ and solve the 1D k-means problem for the Beta distribution above to get centroids $c_1, \ldots, c_{2^b}$. This is done once.
2. At quantization time, rotate: $\mathbf{y} \leftarrow \boldsymbol{\Pi} \cdot \mathbf{x}$.
3. For each coordinate $j$, store $\text{idx}_j = \arg\min_k |\mathbf{y}_j - c_k|$.
4. At dequantization, retrieve the centroids and rotate back via $\boldsymbol{\Pi}^\top$.

We prove a tight distortion bound:

**Theorem 1.** For any bit-width $b \ge 1$ and any $\mathbf{x} \in \mathbb{S}^{d-1}$:

$$D_{\tt mse} \le \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b}$$

At small bit-widths the numerical values are: $D_{\tt mse} \approx 0.36,\ 0.117,\ 0.03,\ 0.009$ for $b = 1, 2, 3, 4$ respectively.

---

## Why MSE-optimal ≠ inner product unbiased

Here's an important subtlety. At 1-bit, the MSE-optimal quantizer maps to signs: $Q_{\tt mse}(\mathbf{x}) = \text{sign}(\boldsymbol{\Pi} \cdot \mathbf{x})$. When you use this for inner product estimation, you get:

$$\mathbb{E}\!\left[ \langle \mathbf{y},\, Q_{\tt mse}^{-1}(Q_{\tt mse}(\mathbf{x})) \rangle \right] = \frac{2}{\pi} \cdot \langle \mathbf{y}, \mathbf{x} \rangle$$

That factor of $2/\pi \approx 0.637$ is a multiplicative bias that cannot be calibrated away without knowing the true inner product. The bias diminishes as $b$ increases — but it's significant at low bit-widths, precisely where quantization matters most.

---

## Inner product CortexQuant: a two-stage fix

The solution is elegant. Rather than trying to fix the bias directly, we use a two-stage approach:

1. Apply $Q_{\tt mse}$ with bit-width $b-1$. This minimizes $\|\mathbf{r}\|_2$ where $\mathbf{r} = \mathbf{x} - Q_{\tt mse}^{-1}(Q_{\tt mse}(\mathbf{x}))$ is the residual.
2. Apply a 1-bit QJL transform on the residual: $\text{qjl} \leftarrow \text{sign}(\mathbf{S} \cdot \mathbf{r})$, where $\mathbf{S} \sim \mathcal{N}(0,1)$ is a random Gaussian matrix.
3. Dequantize as: $\tilde{\mathbf{x}} = \tilde{\mathbf{x}}_{\tt mse} + \frac{\sqrt{\pi/2}}{d} \cdot \|\mathbf{r}\|_2 \cdot \mathbf{S}^\top \cdot \text{qjl}$.

The QJL step provides an unbiased estimate of the residual inner product (by the Johnson-Lindenstrauss lemma), while the MSE stage ensures the residual is as small as possible. Total bit budget: still $b$ bits per coordinate.

**Theorem 2.** For any $b \ge 1$, any $\mathbf{x} \in \mathbb{S}^{d-1}$, and any $\mathbf{y} \in \mathbb{R}^d$:

$$\mathbb{E}\!\left[ \langle \mathbf{y},\, Q_{\tt prod}^{-1}(Q_{\tt prod}(\mathbf{x})) \rangle \right] = \langle \mathbf{y}, \mathbf{x} \rangle \quad \text{(unbiased)}$$

$$D_{\tt prod} \le \frac{\sqrt{3}\,\pi^2 \cdot \|\mathbf{y}\|_2^2}{d} \cdot \frac{1}{4^b}$$

At small bit-widths: $D_{\tt prod} \approx \frac{1.57}{d},\ \frac{0.56}{d},\ \frac{0.18}{d},\ \frac{0.047}{d}$ for $b = 1, 2, 3, 4$ respectively.

---

## How close to optimal? A formal lower bound

Using Yao's minimax principle combined with Shannon's lower bound, we prove that *no* randomized quantizer can do better than:

**Theorem 3.** For any randomized $Q: \mathbb{S}^{d-1} \to \{0,1\}^{b \cdot d}$, there exist hard inputs $\mathbf{x}, \mathbf{y}$ such that:

$$D_{\tt mse}(Q) \ge \frac{1}{4^b} \qquad D_{\tt prod}(Q) \ge \frac{\|\mathbf{y}\|_2^2}{d} \cdot \frac{1}{4^b}$$

Comparing with our upper bounds, CortexQuant is within a factor of at most $\frac{\sqrt{3}\,\pi}{2} \approx 2.7$ of the information-theoretic optimum — for all bit-widths and all dimensions. At $b = 1$, the gap shrinks to just $\approx 1.45\times$.

> The exponential improvement in bit-width dependence ($1/4^b$ rather than $1/2^b$ or worse) means each additional bit buys you twice as much distortion reduction as competing methods provide.

---

## Experimental results

### KV cache: needle-in-a-haystack

We tested CortexQuant on KV cache compression for `Llama-3.1-8B-Instruct` with context lengths from 4k to 104k tokens, at a 4× compression ratio (25% of full cache):

| Method | Recall score |
|---|---|
| SnapKV | 0.858 |
| PyramidKV | 0.895 |
| KIVI | 0.981 |
| PolarQuant | 0.995 |
| Full precision | 0.997 |
| **CortexQuant (ours)** | **0.997** |

CortexQuant matches full-precision recall exactly, even at 4× compression.

### LongBench end-to-end generation

On LongBench-E with `Llama-3.1-8B-Instruct`, CortexQuant at **3.5 bits** matches the full-cache average score of 50.06. At **2.5 bits** it still achieves 49.44 — marginal degradation while compressing by over 5×. For comparison, KIVI requires 5 bits to reach 50.16 and PolarQuant requires 3.9 bits to reach 49.78.

Our non-integer bit-widths (2.5, 3.5) come from splitting channels into outlier and non-outlier groups and applying two independent CortexQuant instances at different precisions. For example, the 2.5-bit setup uses 3 bits for 32 outlier channels and 2 bits for 96 regular channels: $(32 \times 3 + 96 \times 2) / 128 = 2.5$.

### Nearest neighbor search: indexing time

| Method | d=200 | d=1536 | d=3072 |
|---|---|---|---|
| Product Quantization | 37s | 240s | 494s |
| RabitQ | 597s | 2268s | 3957s |
| **CortexQuant (ours)** | **0.0007s** | **0.0013s** | **0.0021s** |

CortexQuant requires essentially zero indexing time — there are no codebooks to train. Despite this, it consistently outperforms both PQ and RabitQ in top-$k$ recall across DBpedia (1536-d, 3072-d) and GloVe (200-d) benchmarks.

---

## Why this matters

The core contribution is not just empirical performance — it's the provable guarantee. For applications like KV cache quantization, where you cannot run offline calibration on the incoming token stream, having a quantizer that is simultaneously **online**, **GPU-friendly**, **unbiased**, and **within 2.7× of the fundamental limit** is qualitatively new.

Entropy coding of the codebook pointers can further reduce the effective bit-width at no distortion cost — though the gain is modest enough (~5% at $b=4$) that we keep the algorithm simple and fast by default.

---

*Full proofs, algorithms, and experimental details are available in the paper. Code release coming soon.*

*Tags: Vector Quantization · KV Cache · LLM Inference · Nearest Neighbor Search · Information Theory*
