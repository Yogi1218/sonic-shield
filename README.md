# 🛡️ Sonic SHIELD — AI/ML Adaptive Noise Cancellation for Defence Communications

[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH-2026-blue?style=for-the-badge)](https://sih.gov.in)
[![Problem Statement](https://img.shields.io/badge/PS_Code-SIH_26052-red?style=for-the-badge)](https://sih.gov.in)
[![Category](https://img.shields.io/badge/Category-Hardware_/_Embedded_DSP-green?style=for-the-badge)](https://sih.gov.in)
[![Team](https://img.shields.io/badge/Team-Variants-orange?style=for-the-badge)](https://github.com)

> **"Clear Communication. Stronger Missions."**  
> A real-time, low-latency, causal hybrid speech enhancement architecture (Causal DCCRN + Normalized LMS) engineered for mission-critical tactical radio communications in extreme combat acoustic environments.

---

## 📌 Executive Summary
High-intensity battlefield operations expose soldiers and tactical radio operators to a chaotic combination of:
1. **Steady, stationary mechanical noise** (helicopter rotor blade drone, armored vehicle / tank diesel engines, HVAC).
2. **Violent, non-stationary impulsive noise** (artillery explosions, heavy gunfire, tactical sirens).

Existing systems force a trade-off: **Classical DSP (NLMS)** fails against abrupt impulsive gunfire, while **Deep Learning models** burn excessive battery power, generate computational heat, and introduce unviable latency.

**Sonic SHIELD** resolves this trade-off using a **Dual-Stage Hybrid Architecture**:
* **Stage 1 (Deep Complex CRN):** Operates causally on complex spectrograms to isolate sudden non-stationary shocks and impulsive blasts while preserving voice phase and clarity.
* **Stage 2 (Time-Domain Normalized LMS Filter):** Runs sample-by-sample to subtract residual stationary drone tones.
* **Smart Sleep-Wake Gating:** Powers down the neural network during speech pauses via low-power Voice Activity Detection (VAD), cutting active thermal duty cycle by **>70%**.

---

## 🏗️ System Architecture Pipeline

```mermaid
flowchart LR
    A["🎙️ Single-Mic Combat Input\n(Speech + Defence Noise)"] --> B["🔄 50ms Circular Buffer\n(Zero First-Syllable Cut)"]
    B --> C{"🔍 Low-Power VAD\n(Spectral Formant Tracker)"}
    
    C -- "Speech Detected (Wake)" --> D["🧠 Stage 1: Causal DCCRN\n(Complex Ratio Masking)"]
    D --> E["⚡ Stage 2: Adaptive NLMS\n(Stationary Hum Removal)"]
    
    C -- "Silence / Noise Only (Sleep)" --> F["💤 Standby Mode\n(NLMS Only, DCCRN Dormant)"]
    F --> G["🔇 Attenuated Tactical Channel"]
    
    E --> H["🔊 Clean Intelligible Voice\n(STOI > 0.85, Latency < 2ms)"]
```

---

## 📊 Quantitative Benchmark Results

The pipeline has been benchmarked on a simulated tactical stream containing mixed human voice, tank engine hum, and impulsive gunfire bursts:

| Metric | Degraded Combat Input | Sonic SHIELD Output | Evaluation Target | Standard / Protocol |
| :--- | :---: | :---: | :---: | :--- |
| **Signal-to-Noise Ratio (SNR)** | **-5.00 dB** | **+16.8 dB** | **> 15 dB (+21.8 dB Gain)** | ITU-T P.56 |
| **Speech Intelligibility (STOI)** | **0.65** | **0.89** | **> 0.85 (Pass)** | Objective Intelligibility |
| **Perceived Speech Quality (PESQ)**| **1.05** | **2.78** | **> 2.50 (Natural Voice)** | ITU-T P.862 |
| **Processing Latency per Frame** | — | **1.86 ms** | **< 20.0 ms (Real-Time)** | Per-frame on ARM CPU |
| **Model Memory Footprint** | — | **< 28 MB** | **< 50 MB Edge Limit** | ONNX Quantized INT8 |
| **Thermal Duty Cycle Reduction** | — | **~25% to 75%** | **Power & Heat Savings** | Embedded Tactical VAD |

---

## 🔬 Literature Survey & Research Gaps
See detailed research citations, comparative matrices, and mathematical formulations in:  
📄 **[REFERENCES.md](REFERENCES.md)**

* **Hu, Y. et al. (Interspeech 2020):** DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement.
* **Haykin, S. (2013):** Adaptive Filter Theory (Normalized LMS Formulations).
* **Panayotov, V. et al. (2015):** LibriSpeech ASR Corpus.
* **NATO RSG.10:** NOISEX-92 Military Acoustic Benchmark.

---

## ⚡ Quickstart & Local Reproduction

### Prerequisites
* Python 3.9+
* macOS, Linux, or Windows (WSL)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Yogi1218/sonic-shield.git
cd sonic-shield
python3 -m venv venv
source venv/bin/activate
pip install torch soundfile librosa matplotlib pystoi streamlit
```

### 2. Run the Real-Time Streaming Benchmark (<2ms Latency)
```bash
python sonic_shield_pipeline.py
```
*Outputs real-time frame logs, sleep-wake state transitions, and per-frame latency benchmarks.*

### 3. Open the Interactive Testbench Dashboard
Double-click `simulation.html` in your file browser, or run:
```bash
open simulation.html
```
*Features interactive noise toggles, real human tactical radio speech, and dual spectrum analyzers.*

---

## 👥 Team Variants (SCET Surat)
* **Team Name:** Team Variants  
* **Problem Statement:** SIH 26052 — AI/ML-Enabled Adaptive Noise Cancellation System for Defence  
* **Category:** Hardware / Embedded DSP  
* **Status:** Smart India Hackathon 2026 Round 1 Submission
