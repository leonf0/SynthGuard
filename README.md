# SynthGuard

## A Regime-Conditioned AI-Driven Protection System Against Unrealistic Synthetic Returns

---

## Project Overview

This project tackles the problem of evaluating whether a synthetic financial return series is *realistic*. A generator can almost perfectly fit the unconditional return distribution yet fail to reproduce the volatility clustering defining a crisis regime, the leverage asymmetry in trending markets, or the autocorrelation decay structure that distinguishes real data from noise. 

SynthGuard serves as a benchmarking pipeline that addresses this directly. Given a candidate time series, it could be produced by any generative model of your choice, the auditor segments it into latent market regimes using a Gaussian Mixture Model and a Markov Chain transition model. Each regime is then evaluated independently against a collection of statistical tests covering canonical stylized facts of financial returns: fat tails, volatility clustering, the leverage effect, gain/loss asymmetry, and coarse-fine volatility correlation etc. A TCN-based discriminator ensemble will then provides a complementary model-based realism score corresponding to the probability that the data is synthetic, a realistic synthetic return sequence should be statistically indistinguishable to real returns and go undetected by this ensemble.

The project comes along with six built-in benchmark generators, three classical parametric models and three deep learning architectures, which serve both as baselines and as worked examples of what the auditor is designed to evaluate.

---

## Section 1 — Regime Fitting and Sequencing

### 1.1 Motivation for Regime Classification

Financial return series are non-stationary. The distributional properties of daily equity returns during a low-volatility market are unlike those of the same instrument during a liquidity crisis. Treating a return series as a homogeneous process and applying stylized fact tests unconditionally produces poor results in both directions: a generator that correctly captures crisis-regime tail behaviour may be penalised because that behaviour is diluted by calmer periods in the aggregate statistic, while a generator that matches the unconditional distribution but gets regime dynamics entirely wrong may appear adequate.

Regime conditioning solves this by partitioning the series into latent states before testing, so that each stylized fact is evaluated within a statistically coherent window. The two components that accomplish this are a Gaussian Mixture Model, which assigns each observation to a regime, and a Markov Chain, which models how regimes succeed one another in time.

---

### 1.2 Gaussian Mixture Models

#### Theory

A Gaussian Mixture Model (GMM) is a probabilistic model that represents the data-generating distribution as a weighted sum of $K$ Gaussian components:

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $\pi_k \geq 0$ are the mixing weights with $\sum_{k=1}^{K} \pi_k = 1$, and each component is parameterised by a mean vector $\boldsymbol{\mu}_k$ and covariance matrix $\boldsymbol{\Sigma}_k$.

The model is trained via the Expectation-Maximisation (EM) algorithm. In the **E-step**, the posterior probability (responsibility) that component $k$ generated observation $\mathbf{x}_i$ is computed:

$$r_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

In the **M-step**, the parameters are updated to maximise the expected complete-data log-likelihood:

$$\pi_k^{\text{new}} = \frac{1}{N} \sum_{i=1}^{N} r_{ik}, \quad \boldsymbol{\mu}_k^{\text{new}} = \frac{\sum_i r_{ik} \mathbf{x}_i}{\sum_i r_{ik}}, \quad \boldsymbol{\Sigma}_k^{\text{new}} = \frac{\sum_i r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^\top}{\sum_i r_{ik}}$$

The E and M steps alternate until the change in log-likelihood falls below a convergence threshold.

<p align="center">
  <img src="assets/diagram_gmm.png" alt="GMM Architecture" width="80%"/>
</p>

#### Training and Fitting

The feature space presented to the GMM is constructed from a rolling window over the input return series. For every time step $t$, the feature vector $\mathbf{x}_t$ encodes: the rolling mean return, rolling volatility (standard deviation of returns), rolling skewness, rolling excess kurtosis, and a measure of short-lag autocorrelation in squared returns as a proxy for volatility clustering intensity. 

Covariance structure is constrained to full matrices to capture inter-feature correlations but regularised with a small diagonal prior to prevent degeneracy on low-variance components.

#### Role in This Project

Once fitted, the GMM assigns each time step a hard regime label (via argmax) and a soft responsibility vector. Hard labels are used to partition the return series for per-regime stylized fact testing. Soft labels are carried forward to the Markov Chain fitting step. The GMM is fitted jointly on VIX data from 2005 to 2025.

---

### 1.3 Markov Chains

#### Theory

A first-order discrete-time Markov Chain models a sequence of states $\{s_t\}_{t=1}^{T}$, $s_t \in \{1, \ldots, K\}$, under the assumption that the future is conditionally independent of the past given the present:

$$P(s_{t+1} = j \mid s_t = i, s_{t-1}, \ldots) = P(s_{t+1} = j \mid s_t = i) = A_{ij}$$

The model is fully specified by its **transition matrix** $\mathbf{A} \in [0,1]^{K \times K}$, where $A_{ij}$ is the probability of transitioning from regime $i$ to regime $j$, and each row sums to one: $\sum_j A_{ij} = 1$.

The stationary distribution $\boldsymbol{\pi}^*$ satisfies $\boldsymbol{\pi}^{*\top} \mathbf{A} = \boldsymbol{\pi}^{*\top}$ and describes the long-run fraction of time spent in each regime. For an ergodic chain, the expected holding time in regime $i$ is:

$$\mathbb{E}[\text{holding time in } i] = \frac{1}{1 - A_{ii}}$$

<p align="center">
  <img src="assets/diagram_markov.png" alt="Markov Chain Architecture" width="80%"/>
</p>

#### Training and Fitting

The transition matrix is calibrated using the hard regime label sequence produced by the GMM using maximum likelihood, which reduces to counting transitions:

$$\hat{A}_{ij} = \frac{n_{ij}}{\sum_{j'} n_{ij'}}$$

where $n_{ij}$ is the number of observed transitions from regime $i$ to regime $j$ in the labelled sequence.

The fitted transition matrix characterises the temporal dynamics of regime sequencing, how long the series tends to dwell in each regime and how likely it is to switch.

---

## Section 2 — Synthetic Generative Models

### 2.1 Parametric Models

The three parametric models serve as interpretable baselines. Each has a known closed form, is fast to fit, and makes specific assumptions about return dynamics that allow controlled ablation of which stylized facts they capture and which they violate by construction.

---

#### 2.1.1 Geometric Brownian Motion (GBM)

GBM is the canonical continuous-time model for asset prices, underpinning Black-Scholes option pricing. Log-returns are assumed to be i.i.d. Gaussian:

$$\ln \frac{S_{t+1}}{S_t} = \mu \Delta t + \sigma \sqrt{\Delta t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$$

GBM has no memory, no volatility clustering, and no fat tails. It is included as the weakest baseline — a generator that should fail virtually every stylized fact test beyond first-moment matching.

**Parametrisation:** $\mu$ and $\sigma$ are estimated from the sample mean and standard deviation of log-returns on the reference series, fit per regime.

---

#### 2.1.2 GARCH(1,1)

The Generalised Autoregressive Conditional Heteroskedasticity model captures volatility clustering by making the conditional variance a function of past squared residuals and past variance:

$$r_t = \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)$$
$$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$$

with the stationarity constraint $\alpha + \beta < 1$. GARCH reproduces volatility clustering and conditionally fat tails but assumes symmetric shocks, so it cannot reproduce the leverage effect.

**Parametrisation:** $(\omega, \alpha, \beta)$ are estimated by maximising the conditional log-likelihood via numerical optimisation (L-BFGS-B), fit per regime.

---

#### 2.1.3 Heston Stochastic Volatility Model

The Heston model allows the variance process to evolve stochastically and independently of the return process, with a correlation between the two Brownian drivers that produces the leverage effect:

$$dS_t = \mu S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^S$$
$$dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_t^v$$
$$dW_t^S \, dW_t^v = \rho \, dt$$

where $\kappa$ is the mean-reversion speed, $\theta$ the long-run variance, $\xi$ the volatility of volatility, and $\rho$ the leverage correlation. The Feller condition $2\kappa\theta > \xi^2$ ensures the variance process remains positive.

**Parametrisation:** Parameters $(\mu, \kappa, \theta, \xi, \rho)$ are estimated per regime by matching empirical moments (variance, autocorrelation of squared returns, skewness) using a method-of-moments procedure with a least-squares objective.

---

### 2.2 Deep Learning Generative Models

---

#### 2.2.1 Temporal Fusion Transformer (TFT)

##### Theory

The Temporal Fusion Transformer is a multi-horizon time series model that combines several architectural components to handle heterogeneous inputs, capture both long and short-range temporal dependencies, and produce attention-based forecasts. Unlike recurrent-only architectures, TFT uses gating mechanisms to selectively suppress irrelevant components of the input, making it robust to variable-length conditioning contexts.

The model processes three categories of input: past observed covariates, known future inputs (such as regime labels fed as conditioning), and static metadata. These are embedded through variable selection networks — feed-forward sub-networks with softmax gates — that learn which inputs are informative for each prediction step.

Temporal dependencies are captured at two scales: a sequence of **LSTM encoders** processes local history, while a **multi-head self-attention** layer over the encoded sequence captures longer-range dependencies. Gated Residual Networks (GRNs) are used throughout as the core transformation block:

$$\text{GRN}(\mathbf{x}) = \text{LayerNorm}\!\left(\mathbf{x} + \text{GLU}\!\left(\mathbf{W}_1 \, \text{ELU}(\mathbf{W}_2 \mathbf{x} + \mathbf{b}_2) + \mathbf{b}_1\right)\right)$$

where GLU is a Gated Linear Unit providing multiplicative suppression.

<p align="center">
  <img src="assets/diagram_tft.png" alt="Temporal Fusion Transformer Architecture" width="80%"/>
</p>

**Architecture Components:**
- **Variable Selection Networks:** Per-timestep softmax-gated feature weighting, one network for static inputs and one for temporal inputs. Learned jointly with the rest of the model.
- **Gated Residual Network (GRN):** Core non-linear transformation block used in variable selection, temporal processing, and output projection. Uses ELU activation followed by a gated linear unit and a residual skip with layer normalisation.
- **LSTM Encoder-Decoder:** Sequence-to-sequence LSTM that encodes past context and initialises the decoder for the prediction horizon.
- **Multi-Head Attention:** Standard scaled dot-product attention over the LSTM output sequence, enabling the model to attend selectively to distant time steps.
- **Quantile Output Heads:** Multiple linear heads producing quantile forecasts at specified probability levels, supporting distributional generation via quantile regression.

##### Training and Inference

The TFT is trained in a teacher-forced autoregressive regime: at each step, the ground truth lagged return is provided as input rather than the model's own previous output. The loss is a pinball (quantile) loss summed across quantile levels:

$$\mathcal{L} = \sum_{q \in \mathcal{Q}} \sum_t \max\!\left(q(y_t - \hat{y}_{t,q}),\, (q-1)(y_t - \hat{y}_{t,q})\right)$$

Regime labels from the GMM are injected as static categorical conditioning inputs, allowing the model to condition its generative distribution on the current regime.

At inference time, returns are generated autoregressively: a return is sampled from the predicted quantile distribution via linear interpolation, fed back as input, and the process repeated for the desired horizon.

---

#### 2.2.2 Conditional Variational Autoencoder (CVAE)

##### Theory

A Variational Autoencoder (VAE) is a latent-variable generative model that learns to encode observations into a structured latent space and decode samples from that space back into the data domain. It optimises a lower bound on the log-likelihood (the ELBO):

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right] - D_{\text{KL}}\!\left(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\right)$$

The encoder $q_\phi(\mathbf{z}|\mathbf{x})$ approximates the posterior over latent variables; the decoder $p_\theta(\mathbf{x}|\mathbf{z})$ models the likelihood. The reparameterisation trick $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ makes the sampling step differentiable.

In the **Conditional** variant, both encoder and decoder receive a conditioning signal $\mathbf{c}$ (here, the regime label and its associated GMM statistics):

$$q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{c}), \quad p_\theta(\mathbf{x}|\mathbf{z}, \mathbf{c})$$

<p align="center">
  <img src="assets/diagram_cvae.png" alt="CVAE Architecture" width="80%"/>
</p>

**Architecture Components:**
- **Encoder:** A stack of 1D convolutional layers followed by a bidirectional GRU that processes an input window of returns. The final hidden state is projected to the mean $\boldsymbol{\mu}_\phi$ and log-variance $\log \boldsymbol{\sigma}^2_\phi$ of the posterior.
- **Conditioning Injection:** The regime label is embedded as a learned dense vector and concatenated to both the encoder input at every timestep and to the decoder input at every step.
- **Latent Bottleneck:** A fixed-dimensional Gaussian latent space. KL weight is annealed from 0 to 1 over the first training phase using a cyclical annealing schedule to prevent posterior collapse.
- **Decoder:** A GRU-based autoregressive decoder that takes the sampled latent $\mathbf{z}$ concatenated with the conditioning vector and generates returns step-by-step.
- **Output Head:** A linear layer projecting the decoder hidden state to return mean and variance, producing a Gaussian output distribution per step.

##### Training and Inference

Training minimises the conditional ELBO with cyclical KL annealing. A free-bits heuristic is applied per latent dimension to prevent the KL term from being trivially satisfied by encoding no information.

At inference, a latent vector is sampled from the prior $\mathcal{N}(0, \mathbf{I})$, concatenated with the regime conditioning vector, and passed to the decoder, which generates a return sequence autoregressively.

---

#### 2.2.3 Score-Based Diffusion Model

##### Theory

Score-based diffusion models learn to generate data by reversing a forward noising process. In the forward process, data $\mathbf{x}_0$ is progressively corrupted by Gaussian noise across $T$ timesteps according to a variance schedule $\{\beta_t\}_{t=1}^T$:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\, \mathbf{x}_{t-1},\, \beta_t \mathbf{I})$$

By the Markov property, the marginal has a closed form:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0,\, (1 - \bar{\alpha}_t)\mathbf{I}), \quad \bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$$

A neural network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ is trained to predict the noise $\boldsymbol{\epsilon}$ added at each step, via the simplified DDPM objective:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\, \boldsymbol{\epsilon},\, t\right)\right\|^2\right]$$

Sampling proceeds by iteratively denoising from $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ using the learned score.

<p align="center">
  <img src="assets/diagram_diffusion.png" alt="Score-Based Diffusion Architecture" width="80%"/>
</p>

**Architecture Components:**
- **U-Net Backbone (1D):** A 1D convolutional U-Net with downsampling and upsampling paths connected by skip connections. Each resolution level consists of two residual convolution blocks.
- **Diffusion Timestep Embedding:** The scalar diffusion timestep $t$ is encoded via a sinusoidal positional embedding (analogous to the Transformer encoding) and projected to a conditioning vector that is added to residual block hidden states via FiLM-style scale-and-shift.
- **Regime Conditioning:** The regime label is embedded as a dense vector and fused with the timestep embedding before injection, enabling the denoising network to learn regime-specific score functions.
- **Attention in Bottleneck:** A single multi-head self-attention layer at the lowest-resolution bottleneck captures long-range structure in the return sequence.
- **Output Head:** A final 1D convolution projecting to the noise prediction $\hat{\boldsymbol{\epsilon}}$ at the input resolution.

##### Training and Inference

Training uses the DDPM objective with a cosine variance schedule. The model is conditioned on regime labels, enabling class-conditional generation. Classifier-free guidance is implemented by randomly dropping the regime conditioning with probability 0.1 during training, allowing guidance strength to be tuned at inference.

At inference, a pure noise sequence is sampled and progressively denoised over $T$ reverse steps using the DDIM sampler for accelerated generation, with the regime label injected at every denoising step.

---

## Section 3 — TCN Discriminator Ensemble

### 3.1 Theory of Temporal Convolutional Networks

A Temporal Convolutional Network (TCN) is a class of 1D convolutional architecture designed for sequence modelling. The two properties that distinguish it from a standard CNN are **causal convolutions** and **dilated convolutions**.

A causal convolution ensures that the output at time $t$ depends only on inputs at times $\leq t$, preventing information leakage from the future. This is implemented by padding the input on the left by $(k-1)$ zeros, where $k$ is the kernel size, and applying a standard 1D convolution.

Dilated convolutions introduce gaps of size $d$ between kernel elements, exponentially expanding the receptive field without increasing the number of parameters:

$$(\mathbf{W} *_d \mathbf{x})(t) = \sum_{i=0}^{k-1} \mathbf{W}(i) \cdot \mathbf{x}(t - d \cdot i)$$

By stacking layers with dilation factors $d = 1, 2, 4, \ldots, 2^{L-1}$, a TCN with $L$ layers and kernel size $k$ achieves a receptive field of size $(k-1) \cdot 2^L + 1$, covering long history with $O(\log N)$ layers.

Each TCN layer is wrapped in a **residual block** with weight normalisation:

$$\mathbf{h}^{(l)} = \text{ReLU}\!\left(\mathbf{x}^{(l)} + \text{Conv}_{d_l}\!\left(\text{Dropout}\!\left(\text{Conv}_{d_l}(\mathbf{x}^{(l)})\right)\right)\right)$$

with a $1\times 1$ convolution on the skip path when input and output channel dimensions differ.

<p align="center">
  <img src="assets/diagram_tcn.png" alt="TCN Discriminator Ensemble Architecture" width="80%"/>
</p>

**Architecture Components:**
- **Input Projection:** A $1\times 1$ convolution mapping the raw return sequence (single channel) to the model's hidden dimension.
- **Dilated Residual Stack:** A sequence of causal dilated residual blocks with exponentially increasing dilation factors. Each block contains two weight-normalised causal convolutions, a ReLU non-linearity, and spatial dropout between the two convolution layers.
- **Regime Conditioning:** The GMM-assigned regime label is embedded as a dense vector and added to the hidden state at each residual block via a learned affine transformation (FiLM), conditioning the discriminator's judgement on the current regime.
- **Global Aggregation:** After the dilated stack, a global average pooling operation collapses the temporal dimension to a fixed-length representation.
- **Classification Head:** A two-layer MLP projecting the pooled representation to a scalar logit representing the probability that the input sequence is real.

### 3.2 Ensemble Training

The discriminator is trained as a binary classifier: real reference windows are labelled 0, synthetic windows from each generator are labelled 1. Training uses the binary cross-entropy loss:

$$\mathcal{L}_{\text{disc}} = -\mathbb{E}_{\mathbf{x} \sim p_{\text{real}}}\!\left[\log D(\mathbf{x})\right] - \mathbb{E}_{\mathbf{x} \sim p_{\text{synth}}}\!\left[\log(1 - D(\mathbf{x}))\right]$$

The ensemble consists of $M$ independently initialised and trained TCNs. Each member is trained on a different bootstrap resample of the training windows, introducing diversity through data perturbation rather than architectural variation. Class imbalance between the single real series and multiple synthetic generators is addressed by upsampling real windows to match the total synthetic count.

### 3.3 Inference

At inference, a candidate return series is segmented into overlapping windows. Each window is passed through all $M$ ensemble members, producing $M$ real/synthetic probability estimates. The ensemble prediction is the mean probability across members, and uncertainty is quantified as the standard deviation. The realism score reported by the auditor is the mean ensemble probability that the series is real, broken down per regime by filtering windows to those assigned each regime label by the GMM.

---

## Section 4 — Statistical Test Suite

### 4.1 Fat Tails (Excess Kurtosis and Hill Estimator)

**Intuition:** Real financial returns have heavier tails than a Gaussian distribution — large daily moves occur far more frequently than the normal distribution predicts. This test checks whether the synthetic series reproduces this property.

**Theory:** Excess kurtosis is defined as $\kappa = \mathbb{E}[(r - \mu)^4]/\sigma^4 - 3$. For a Gaussian, $\kappa = 0$; empirical equity returns typically show $\kappa \in [2, 10]$. The test statistic compares $\hat{\kappa}_{\text{synth}}$ against the reference distribution of $\hat{\kappa}_{\text{real}}$ via a bootstrap confidence interval.

For the tail specifically, the Hill estimator provides a non-parametric estimate of the tail index $\alpha$ of a regularly varying distribution $P(|r| > x) \sim x^{-\alpha}$:

$$\hat{\alpha}_{\text{Hill}} = \left(\frac{1}{k} \sum_{i=1}^{k} \ln \frac{r_{(n-i+1)}}{r_{(n-k)}}\right)^{-1}$$

where $r_{(1)} \leq \cdots \leq r_{(n)}$ are order statistics and $k$ is the number of upper-order statistics used. Heavier tails correspond to smaller $\alpha$. Empirical equity returns typically yield $\hat{\alpha} \in [3, 5]$. The test flags series whose $\hat{\alpha}$ lies outside a bootstrap confidence interval around the reference estimate.

---

### 4.2 Absence of Autocorrelation in Raw Returns

**Intuition:** While asset prices are not predictable (by the efficient market hypothesis), the direction of this test is that raw returns should show no significant linear autocorrelation — a generator that produces serially correlated returns is implicitly claiming predictable price movements.

**Theory:** The Ljung-Box test evaluates the joint null hypothesis that the first $h$ autocorrelations of the return series are all zero:

$$Q_{\text{LB}} = T(T+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{T-k} \xrightarrow{d} \chi^2(h)$$

where $\hat{\rho}_k$ is the sample autocorrelation at lag $k$ and $T$ is the series length. The test reports the $p$-value against $H_0$; a rejection (low $p$-value) on the synthetic series indicates spurious autocorrelation not present in the reference.

---

### 4.3 Volatility Clustering

**Intuition:** Even if returns are not autocorrelated, large moves tend to cluster together — a turbulent day is more likely to be followed by another turbulent day than by a calm one. This is the defining feature of GARCH-type dynamics and is captured by significant autocorrelation in the magnitude or squared returns.

**Theory:** The autocorrelation function of absolute returns $|r_t|$ and squared returns $r_t^2$ is computed up to a maximum lag $h$. Significant positive autocorrelation at short lags (typically decaying slowly over lags 1–20) constitutes evidence of volatility clustering. The test uses the Ljung-Box statistic on $r_t^2$ and compares the autocorrelation profile of the synthetic series against the reference using an $\ell_1$ distance over the ACF vector. A permutation test generates the null distribution of this distance under the hypothesis that the synthetic series has the same ACF structure as the reference.

---

### 4.4 Leverage Effect

**Intuition:** In equity markets, volatility tends to increase more after negative returns than after positive returns of the same magnitude — a crash is followed by more turbulence than a rally. This asymmetry is the leverage effect, and it is absent from symmetric models like GARCH with Gaussian innovations.

**Theory:** The leverage effect is measured via the cross-correlation between past returns and future squared returns:

$$\mathcal{L}(k) = \text{Corr}(r_t^2, r_{t-k}), \quad k > 0$$

For real equity returns, $\mathcal{L}(k) < 0$ for small positive lags $k$ — negative returns at $t-k$ predict higher future volatility at $t$. A symmetric generative model produces $\mathcal{L}(k) \approx 0$. The test computes $\hat{\mathcal{L}}(k)$ for both reference and synthetic series at lags $k = 1, \ldots, 10$ and applies a Fisher $z$-test to compare the two cross-correlation profiles. The sign, magnitude, and decay rate of the leverage profile are all reported.

---

### 4.5 Gain/Loss Asymmetry

**Intuition:** Markets fall faster than they rise. Large drawdowns happen over shorter horizons than equivalent rallies, and the distribution of negative returns is more extreme than its positive counterpart. A generator that is symmetric about zero fails this test.

**Theory:** Gain/loss asymmetry is operationalised via two complementary statistics. First, the ratio of downside semi-deviation to upside semi-deviation:

$$\text{GLA} = \frac{\sqrt{\mathbb{E}[\min(r_t, 0)^2]}}{\sqrt{\mathbb{E}[\max(r_t, 0)^2]}}$$

Values greater than 1 indicate heavier downside. Second, the conditional tail ratio compares the $5\%$ and $95\%$ quantiles: $|Q_{0.05}| / Q_{0.95}$. Both statistics are computed for the synthetic and reference series and compared using bootstrap confidence intervals.

---

### 4.6 Coarse-Fine Volatility Correlation (Taylor Effect)

**Intuition:** Volatility estimated at a coarse time scale (e.g., weekly) is positively correlated with volatility estimated at a fine scale (e.g., daily) from the same period. This multi-scale coherence is a non-trivial property of real return series that naive generators often fail to reproduce.

**Theory:** Coarse volatility $\sigma^{\text{coarse}}_t$ is computed as the standard deviation of returns within non-overlapping windows of length $W_c$ (e.g., 20 days). Fine volatility $\sigma^{\text{fine}}_t$ is the standard deviation within sub-windows of length $W_f < W_c$ (e.g., 5 days). The coarse-fine correlation is:

$$\rho_{\text{CF}} = \text{Corr}\!\left(\sigma^{\text{coarse}}_t, \sigma^{\text{fine}}_t\right)$$

The Taylor effect additionally refers to the empirical observation that $\text{Corr}(|r_t|, |r_{t+k}|)$ decays more slowly than $\text{Corr}(r_t^2, r_{t+k}^2)$, indicating that absolute returns are more persistent than squared returns. The test computes both the coarse-fine correlation and the ratio of ACF decay rates for $|r_t|$ versus $r_t^2$, comparing both against the reference series via bootstrap.

---

### 4.7 Aggregational Gaussianity

**Intuition:** While daily returns have fat tails, returns aggregated over longer horizons (weekly, monthly) progressively approach a Gaussian distribution by the Central Limit Theorem. A well-calibrated generator should reproduce this convergence — fat tails at fine scales and near-Gaussian behaviour at coarse scales.

**Theory:** Returns are aggregated at horizons $h \in \{1, 5, 21, 63\}$ trading days by summing over non-overlapping windows. At each horizon, the Jarque-Bera test statistic is computed:

$$\text{JB} = \frac{T}{6}\!\left(\hat{S}^2 + \frac{(\hat{\kappa})^2}{4}\right)$$

where $\hat{S}$ is sample skewness and $\hat{\kappa}$ excess kurtosis. The test plots the JB statistic as a function of aggregation horizon for both real and synthetic series and measures the rate at which the statistic decays toward zero. A generator that fails aggregational Gaussianity either retains fat tails at long horizons (over-heavy tails) or converges too quickly (under-heavy tails at short horizons). The convergence profiles are compared using a curve-distance metric on the JB-vs-horizon series.

---

### 4.8 Hurst Exponent (Long Memory)

**Intuition:** Real volatility series exhibit long-range dependence — autocorrelations in absolute returns decay slowly as a power law rather than exponentially. This long memory means that shocks to volatility persist over much longer horizons than a simple ARMA model would predict. A generator without long-memory dynamics will show autocorrelations that decay too quickly.

**Theory:** The Hurst exponent $H$ characterises the self-similarity of a time series. For a fractionally integrated process, $H \in (0.5, 1)$ indicates positive long-range dependence. It is estimated via rescaled range (R/S) analysis:

$$\frac{R(n)}{S(n)} \sim c \cdot n^H$$

where $R(n) = \max_{1 \leq k \leq n} \sum_{t=1}^k (r_t - \bar{r}) - \min_{1 \leq k \leq n} \sum_{t=1}^k (r_t - \bar{r})$ is the range of cumulative deviations and $S(n)$ is the standard deviation over a window of length $n$. A log-log regression of $R(n)/S(n)$ against $n$ over a range of window sizes yields $\hat{H}$. For real equity returns in absolute value, $\hat{H} \approx 0.6$–$0.7$ is typical. The test compares $\hat{H}_{\text{synth}}$ against a bootstrap confidence interval around $\hat{H}_{\text{real}}$, separately for raw returns and for absolute returns.
