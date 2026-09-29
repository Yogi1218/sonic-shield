# Literature Survey and Theoretical Foundations

Smart India Hackathon 2026 | Problem Statement: SIH 26052 | Team PHALANX

---

## 1. Why Existing Solutions Fall Short in Defence Scenarios

When evaluating existing literature for tactical communications in combat environments, standard approaches consistently encounter several fundamental engineering bottlenecks:

1. **The Single-Microphone Barrier:** Most multi-channel beamforming techniques (such as Generalized Sidelobe Cancellers) assume a multi-microphone array spaced along a headset. In field combat, soldiers frequently use simple, rugged single-capsule tactical throat mics or boom mics where spatial filtering cannot be applied.
2. **Phase Neglect in Deep Learning:** Early deep speech enhancement models (like SEGAN or basic UNet masks) operate only on magnitude spectrograms, discarding or reusing the noisy phase. At low signal-to-noise ratios (-5 dB to -10 dB), corrupted phase produces an unnatural, robotic voice and significantly lowers speech intelligibility scores (STOI).
3. **Adaptive Filter Divergence:** Classic time-domain filters like LMS and NLMS track stationary hum very efficiently. However, sudden shockwaves from gunfire or artillery cause the input energy to spike by 30 to 40 dB within a few samples, which can blow up filter weights unless actively protected.
4. **Thermal and Energy Constraints on Edge Hardware:** Running a deep complex neural network continuously on an infantry soldier's battery-powered radio drains the pack quickly and generates excessive heat inside an enclosed enclosure.

Sonic SHIELD addresses these limitations by pairing an intelligent low-power voice trigger (TENVAD) with a phase-preserving complex neural network (DCCRN) and an adaptive NLMS post-filter.

---

## 2. Comparative Analysis of Published Literature

| Citation & Authors | Core Architecture | Key Strengths | Practical Limitation in Combat | How We Address It in Sonic SHIELD |
|---|---|---|---|---|
| **Hu, Y. et al. (Interspeech 2020)**<br>*DCCRN: Deep Complex Convolution Recurrent Network* | Complex Conv2D + Complex LSTM using Complex Ratio Masking (CRM) | Preserves both speech magnitude and phase; state-of-the-art speech intelligibility in noisy environments. | Heavy continuous computational load that quickly depletes tactical radio battery life. | We use our TENVAD unit to put the complex network to sleep during pauses, reducing active duty cycle by 25% to 75%. |
| **Haykin, S. (Pearson Education, 2013)**<br>*Adaptive Filter Theory (5th Edition)* | Normalized Least Mean Squares (NLMS) time-domain adaptive filtering | Sub-millisecond execution latency; excellent at tracking continuous periodic engine drones. | Filter weights diverge or destabilize when hit by sudden impulsive gunfire or artillery blasts. | The DCCRN stage absorbs non-stationary and impulsive transients, feeding a conditioned reference into the NLMS stage. |
| **Zheng, C. et al. (AAAI 2021)**<br>*Interactive Speech and Noise Modeling (SN-Net)* | Dual-branch neural network estimating both speech and noise features | Explicitly models noise characteristics rather than treating noise purely as a residual. | Discards the estimated noise branch after inference instead of using it to refine filtering. | We take the secondary noise mask output and route it directly as the physical reference signal for our NLMS post-filter. |
| **Valin, J.-M. (IEEE MMSP 2018)**<br>*Real-Time Speech Enhancement (RNNoise)* | Hybrid GRU predicting 22 Bark-scale bands with a pitch comb filter | Extremely lightweight footprint (<100K weights); runs easily on low-power microcontrollers. | Coarse frequency bands struggle with phase reconstruction in negative SNRs, producing noticeable artifacts. | We use complex spectrogram ratio masking (CRM) across fine frequency bins to maintain vocal formant clarity down to -5 dB SNR. |
| **Fedorov, I. et al. (Interspeech 2020)**<br>*Tiny Recurrent Networks on Microcontrollers* | INT8 quantized causal GRUs deployed on ARM Cortex-M microcontrollers | Fits into small on-chip SRAM with minimal power draw. | Limited network capacity introduces audible musical noise under heavy background machinery hum. | We offload stationary engine hum cancellation to the NLMS filter, allowing a compact neural network to focus solely on speech isolation. |

For detailed breakdowns of all 10 surveyed research publications, see [docs/LITERATURE_SURVEY_ANALYSIS.md](docs/LITERATURE_SURVEY_ANALYSIS.md).

---

## 3. Mathematical Foundations

### 3.1 Short-Time Fourier Transform (STFT) Analysis
Audio is sampled at $F_s = 16\text{ kHz}$ and framed into causal 32 ms windows with 16 ms hops ($N_{\text{fft}} = 512, R = 256$ samples):
$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] \cdot w[n - mR] \cdot e^{-j \omega n}$$

We apply a periodic Hann window $w[n]$ to minimize spectral leakage across neighboring frequency bins.

### 3.2 Complex Ratio Masking (CRM)
Standard real masks only scale spectral magnitudes, leaving the noisy phase unaltered. In the complex domain, the true clean speech spectrum $S = S_r + j S_i$ is estimated by multiplying the noisy complex spectrum $Y = Y_r + j Y_i$ by a complex mask $M = M_r + j M_i$:
$$S_{\text{est}} = M \cdot Y = (M_r Y_r - M_i Y_i) + j (M_r Y_i + M_i Y_r)$$

This formulation enables the network to simultaneously restore speech magnitude while correcting phase distortion caused by background acoustics.

### 3.3 Normalized LMS (NLMS) Weight Adaptation
The time-domain post-filter updates its coefficient vector $\mathbf{w}(n)$ on each incoming sample using the error $e(n)$ and the reference noise vector $\mathbf{x}(n)$:
$$e(n) = d(n) - \mathbf{w}^T(n) \mathbf{x}(n)$$
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu}{\|\mathbf{x}(n)\|^2 + \epsilon} \cdot e(n) \cdot \mathbf{x}(n)$$

Where:
* $\mu$: Adaptation step size (set to $\mu = 0.08$ for steady convergence).
* $\epsilon$: Regularization constant ($\epsilon = 10^{-4}$) to prevent division by zero during silent intervals.
* $\|\mathbf{x}(n)\|^2$: Instantaneous input signal power that normalizes the step size, preventing filter instability during sudden sound level shifts.

To prevent the filter from accidentally cancelling the soldier's own voice (speech self-cancellation), adaptation is halted ($\mu = 0$) whenever the voice activity detector flags active speech.

For a full step-by-step mathematical derivation of the filter coefficients and power normalization, see [docs/NLMS_MATHEMATICAL_MODEL.md](docs/NLMS_MATHEMATICAL_MODEL.md).

---

## 4. Benchmark Datasets and Test Corpora

To evaluate performance objectively, we utilized established open research corpora:

1. **Clean Voice Targets:**
   * **LJ Speech Dataset 1.1:** 13,100 clean single-speaker voice passages sampled at 22.05 kHz (downsampled to 16 kHz for tactical narrow/wideband standard).
   * **LibriSpeech (train-clean):** Public acoustic speech corpus providing varied acoustic phonemes across diverse English speakers.

2. **Defence and Environmental Acoustic Noise:**
   * **Military Acoustic Database (MAD Dataset):** Real-world defence operational noise corpus containing field recordings of jet aircraft (F-22 Raptor), transport helicopters, sirens, armored vehicle engines, and weapon discharge acoustics.
   * **NOISEX-92 (NATO RSG.10):** The definitive military benchmark for steady stationary interference, including tank turret and armored vehicle cabin hum.

---
*Authored and maintained by Team PHALANX for Smart India Hackathon 2026 (Problem Statement SIH 26052).*
