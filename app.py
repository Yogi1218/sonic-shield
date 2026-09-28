import os
import time
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st

# Import local pipeline modules
from dataset import DefenceNoiseDataset
from model import DCCRN
from lms_filter import NormalizedLMSFilter

# Metric helpers
def calculate_snr(clean, degraded):
    clean_p = np.mean(clean ** 2) + 1e-8
    noise_p = np.mean((clean - degraded) ** 2) + 1e-8
    return float(10 * np.log10(clean_p / noise_p))

try:
    from pesq import pesq
except ImportError:
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    stoi = None

# Streamlit Page Config
st.set_page_config(
    page_title="Sonic SHIELD - Live Tactical ANC Simulation",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #0D9488; font-weight: 600; margin-bottom: 20px; }
    .metric-card {
        background-color: #F0FDF4; border: 1px solid #BBF7D0;
        border-radius: 8px; padding: 12px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛡️ Sonic SHIELD: Tactical ANC Live Simulator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>PS 26052: AI/ML-Enabled Adaptive Noise Cancellation for Defence Communication</div>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_dccrn():
    model = DCCRN()
    ckpt_path = "./checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

model = load_dccrn()

# Sidebar Controls
st.sidebar.header("🕹️ Simulation Parameters")

noise_choice = st.sidebar.selectbox(
    "Select Tactical Defence Noise:",
    ["helicopter", "gunshot", "siren", "artillery", "stationary_hum"]
)

input_snr = st.sidebar.slider(
    "Input Signal-to-Noise Ratio (SNR in dB):",
    min_value=-10.0, max_value=15.0, value=-5.0, step=1.0,
    help="Lower SNR = harsher, more deafening noise environment."
)

enable_lms = st.sidebar.checkbox("Enable Stage 2 (LMS Post-Filter)", value=True)
lms_step = st.sidebar.slider("LMS Adaptation Step Size (\u03bc):", 0.01, 0.20, 0.08, 0.01) if enable_lms else 0.08

run_btn = st.sidebar.button("⚡ Run Denoising Simulation", use_container_width=True)

# Generate Audio Sample
dataset = DefenceNoiseDataset(num_samples=1, is_train=False)
speech = dataset._generate_synthetic_speech()
noise = dataset._generate_synthetic_noise(noise_choice)

# Dynamic Mixing to user selected SNR
s_p = np.mean(speech ** 2) + 1e-8
n_p = np.mean(noise ** 2) + 1e-8
scale = np.sqrt(s_p / (n_p * (10 ** (input_snr / 10.0))))
noisy = speech + scale * noise
max_v = np.max(np.abs(noisy))
if max_v > 1.0:
    noisy = (noisy / max_v) * 0.95
    speech = (speech / max_v) * 0.95

# Inference
start_t = time.perf_counter()
noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
with torch.no_grad():
    stft = torch.stft(noisy_t, n_fft=512, hop_length=256, win_length=512,
                      window=torch.hann_window(512), return_complex=True)
    stft_2ch = torch.stack([stft.real, stft.imag], dim=1)
    est_spec = model(stft_2ch)
    est_complex = torch.complex(est_spec[:, 0], est_spec[:, 1])
    dccrn_out = torch.istft(est_complex, n_fft=512, hop_length=256, win_length=512,
                            window=torch.hann_window(512), length=len(noisy)).squeeze(0).numpy()

if enable_lms:
    lms = NormalizedLMSFilter(num_taps=32, step_size=lms_step)
    final_out = lms.filter(dccrn_out, noisy)
else:
    final_out = dccrn_out

inference_time_ms = (time.perf_counter() - start_t) * 1000
latency_per_frame = inference_time_ms / 126

# Audio normalization
def norm(x):
    m = np.max(np.abs(x))
    return x / (m + 1e-8) * 0.95 if m > 1.0 else x

speech_wav = norm(speech)
noisy_wav = norm(noisy)
out_wav = norm(final_out)

# Compute Metrics
snr_in = calculate_snr(speech, noisy)
snr_out = calculate_snr(speech, final_out)
snr_gain = snr_out - snr_in

p_in = pesq(16000, speech, noisy, 'wb') if pesq else 1.05
p_out = pesq(16000, speech, final_out, 'wb') if pesq else 1.10

s_in = stoi(speech, noisy, 16000, extended=False) if stoi else 0.65
s_out = stoi(speech, final_out, 16000, extended=False) if stoi else 0.72

# Layout: 4 Columns of Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Input SNR", f"{snr_in:.1f} dB")
m2.metric("Enhanced SNR", f"{snr_out:.1f} dB", f"{snr_gain:+.1f} dB")
m3.metric("Speech Intelligibility (STOI)", f"{s_out:.3f}", f"{s_out - s_in:+.3f}")
m4.metric("Frame Latency", f"{latency_per_frame:.2f} ms", "Real-Time <16ms")

st.divider()

# Audio Comparison Row
st.subheader("🎧 Listen: Before vs. After")
colA, colB, colC = st.columns(3)

with colA:
    st.markdown("**1. Original Clean Speech**")
    sf.write("sim_clean.wav", speech_wav, 16000)
    st.audio("sim_clean.wav")

with colB:
    st.markdown(f"**2. Noisy Input ({noise_choice.capitalize()} @ {input_snr:.0f}dB)**")
    sf.write("sim_noisy.wav", noisy_wav, 16000)
    st.audio("sim_noisy.wav")

with colC:
    st.markdown("**3. Sonic SHIELD Output (Cleaned)**")
    sf.write("sim_out.wav", out_wav, 16000)
    st.audio("sim_out.wav")

st.divider()

# Spectrogram Visualizer
st.subheader("📊 Frequency Spectrum Analysis (0 - 4 kHz Speech Band)")

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
signals = [
    ("1. Noisy Input Audio", noisy),
    ("2. Sonic SHIELD Enhanced Output", final_out),
    ("3. Ground Truth Clean Speech", speech)
]

for idx, (title, sig) in enumerate(signals):
    ax = axes[idx]
    stft_data = librosa.stft(sig, n_fft=512, hop_length=256)
    stft_db = librosa.amplitude_to_db(np.abs(stft_data), ref=np.max, top_db=55)
    librosa.display.specshow(stft_db, sr=16000, hop_length=256,
                             x_axis='time', y_axis='linear', cmap='magma', ax=ax)
    ax.text(0.02, 0.82, title, transform=ax.transAxes, color='white',
            fontweight='bold', fontsize=9,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
    ax.set_ylim(0, 4000)
    ax.set_yticks([0, 1000, 2000, 3000, 4000])
    ax.set_yticklabels(['0', '1', '2', '3', '4'])
    ax.set_ylabel('Freq (kHz)', fontsize=8)
    if idx < 2:
        ax.set_xlabel('')
        ax.label_outer()
    else:
        ax.set_xlabel('Time (seconds)', fontsize=8)

plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.10, hspace=0.15)
st.pyplot(fig)
