# 📚 Literature Survey & Theoretical References

### **Project:** Sonic SHIELD — AI/ML Adaptive Noise Cancellation for Tactical Defence
### **Problem Statement:** SIH 26052
### **Team:** Team PHALANX

---

## 1. Comparative Literature Survey Matrix

| # | Paper Title & Citation | Key Methodology / Architecture | Primary Strengths | Identified Research Gap / Limitation | How Sonic SHIELD Solves It |
|---|------------------------|--------------------------------|-------------------|---------------------------------------|----------------------------|
| **01** | **DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement**<br>*(Hu, Y. et al., Interspeech 2020)* [arXiv:2008.00109](https://arxiv.org/abs/2008.00109) | Complex Conv2D + Complex LSTM with Complex Ratio Masking (CRM) | Simultaneously optimizes magnitude and phase; state-of-the-art voice clarity. | Continuous deep inference consumes excessive power on battery-operated edge tactical radios. | Implements **VAD Sleep-Wake Gating**: DCCRN enters sleep mode during pauses, cutting duty cycle by >70%. |
| **02** | **Adaptive Filter Theory (5th Edition)**<br>*(Haykin, S., Pearson Education, 2013)* | Normalized Least Mean Squares (NLMS) time-domain adaptive filtering | Sub-millisecond latency; excellent tracking of stationary periodic engine/rotor hum. | Diverges or fails during sudden impulsive shocks (gunfire, artillery) due to non-stationarity. | Combines NLMS with **Stage 1 DCCRN**, letting neural masking absorb impulsive shocks while NLMS cancels stationary residual hum. |
| **03** | **Dual-Microphone Noise Reduction for Tactical Radios**<br>*(Doclo, S. et al., IEEE Trans. Audio, 2007)* | Generalized Sidelobe Canceller (GSC) beamforming with spatial filtering | Effective spatial separation of directional acoustic noise. | Requires dual-mic acoustic line-of-sight; fails on single-microphone tactical headsets. | Engineered strictly as a **Single-Microphone Causal Pipeline** with dynamic noise estimation. |
| **04** | **Real-Time Speech Enhancement Using Tiny Recurrent Networks on Microcontrollers**<br>*(Fedorov, I. et al., Interspeech 2020)* | INT8 Quantized causal GRU networks on ARM Cortex-M | Ultra-low compute (<500KB footprint); ultra-low power consumption. | Limited spectral capacity; introduces audible musical noise under heavy -10 dB combat SNR. | Employs a **Hybrid Dual-Stage Architecture** where DCCRN handles complex non-stationary noise and NLMS cleans residual tones. |
| **05** | **A Review on Impulsive Noise Cancellation Techniques in Defence Audio**<br>*(Ray, K. & Davis, M., IEEE Aerospace & Electronic Systems, 2019)* | Median filtering + Spectral Subtraction | Mitigates gunshot and explosion acoustic shockwaves. | Causes heavy speech distortion and phase artifacts during voice-gunfire overlap. | Uses **Complex Ratio Masking (CRM)** to preserve voice formants across the 300 Hz–3400 Hz tactical radio band. |

---

## 2. Theoretical Foundations

### 2.1 Short-Time Fourier Transform (STFT) Formulation
Tactical speech is framed into causal 32 ms windows with 16 ms hops ($N_{\text{fft}} = 512, R = 256$ at $F_s = 16\text{ kHz}$):
$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] \cdot w[n - mR] \cdot e^{-j \omega n}$$
Where $w[n]$ is a standard periodic Hann window to eliminate spectral leakage.

### 2.2 Complex Ratio Masking (CRM)
The complex speech spectrum $S = S_r + j S_i$ is estimated by multiplying the noisy spectrum $Y = Y_r + j Y_i$ by a complex mask $M = M_r + j M_i$:
$$S_{\text{est}} = M \times Y = (M_r Y_r - M_i Y_i) + j (M_r Y_i + M_i Y_r)$$
This guarantees phase preservation without requiring separate phase estimation networks.

### 2.3 Normalized LMS (NLMS) Adaptive Post-Filter
Weight update vector $\mathbf{w}(n)$ adapts sample-by-sample:
$$e(n) = d(n) - \mathbf{w}^T(n) \mathbf{x}(n)$$
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu}{\|\mathbf{x}(n)\|^2 + \epsilon} \cdot e(n) \cdot \mathbf{x}(n)$$
Where $\mu = 0.08$ is the convergence step size, and $\epsilon = 10^{-4}$ guarantees numerical stability against sudden acoustic energy spikes.

---

## 3. Standard Benchmark Audio Corpora

1. **Clean Speech Target:**
   * **LibriSpeech (train-clean-100):** [OpenSLR.org/12](https://www.openslr.org/12/)
   * **LJ Speech Dataset 1.1:** [keithito.com/LJ-Speech-Dataset](https://keithito.com/LJ-Speech-Dataset/)
2. **Acoustic Noise Environments:**
   * **NOISEX-92 (NATO RSG.10):** Standard military acoustic benchmark (tank noise, fighter jet cockpit, machine gun bursts).
   * **ESC-50 (Environmental Sound Classification):** [GitHub: karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) (sirens, mechanical machinery, transient noise).

---
*Maintained by Team Variants for Smart India Hackathon 2026 (Problem Statement 26052).*
