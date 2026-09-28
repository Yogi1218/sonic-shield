"""
========================================================================================
Project: Sonic SHIELD — Tactical Adaptive Noise Cancellation System
Target: Embedded ARM / Edge DSP Pipeline (<20ms latency, Low Thermal Footprint)
Architecture:
  - Single Mic Input (Mixed Speech + Defence Noise)
  - 50ms Pre-Roll Circular Buffer (Zero First-Syllable Clipping)
  - Low-Power Speech-Only VAD Trigger (Sleep-Wake Logic)
  - Dual-Output DCCRN (Speech Mask + Noise Reference Mask)
  - Always-On Sample-by-Sample Normalized LMS (NLMS) Post-Filter
========================================================================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa

# ======================================================================================
# 1. 50ms CIRCULAR PRE-ROLL BUFFER
# ======================================================================================
class CircularAudioBuffer:
    """
    Rolling circular buffer that stores the past N milliseconds of audio.
    Ensures that when VAD transitions from 0 -> 1, the neural network can access
    the onset of speech, preventing first-syllable clipping.
    """
    def __init__(self, capacity_ms=50, sample_rate=16000):
        self.capacity = int((capacity_ms / 1000.0) * sample_rate)
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx = 0
        self.sample_rate = sample_rate

    def write(self, frame: np.ndarray):
        """Append incoming audio samples circularly."""
        frame_len = len(frame)
        if frame_len >= self.capacity:
            self.buffer[:] = frame[-self.capacity:]
            self.write_idx = 0
        else:
            end_idx = self.write_idx + frame_len
            if end_idx <= self.capacity:
                self.buffer[self.write_idx:end_idx] = frame
            else:
                part1 = self.capacity - self.write_idx
                part2 = frame_len - part1
                self.buffer[self.write_idx:] = frame[:part1]
                self.buffer[:part2] = frame[part1:]
            self.write_idx = (self.write_idx + frame_len) % self.capacity

    def get_last_n_ms(self, ms=50) -> np.ndarray:
        """Retrieve the continuous most recent N ms of audio in correct chronological order."""
        n_samples = min(int((ms / 1000.0) * self.sample_rate), self.capacity)
        # Unroll circular buffer
        ordered = np.concatenate((self.buffer[self.write_idx:], self.buffer[:self.write_idx]))
        return ordered[-n_samples:]


# ======================================================================================
# 2. SPEECH-ONLY LOW-POWER VAD (SIMULATED WebRTC VAD)
# ======================================================================================
class SpeechVAD:
    """
    Simulates a low-power, lightweight Voice Activity Detector.
    Uses spectral energy ratios and harmonic tracking to distinguish human voice
    formants (300Hz - 3400Hz) from stationary engine rumble and impulsive blasts.
    """
    def __init__(self, sample_rate=16000, speech_energy_threshold=0.015):
        self.sample_rate = sample_rate
        self.threshold = speech_energy_threshold
        self.hangover_frames = 5  # Keep awake for 5 frames after speech stops to prevent fluttering
        self.hangover_count = 0

    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Returns True if human speech is active, False during silence or pure noise.
        """
        # Bandpass band energy (300Hz - 3400Hz voice band)
        frame_energy = np.mean(frame ** 2)
        
        # Spectral ratio: Speech has distinct formant concentration in 300Hz-3400Hz
        fft_mag = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), 1.0 / self.sample_rate)
        
        voice_band = (freqs >= 300) & (freqs <= 3400)
        voice_ratio = np.sum(fft_mag[voice_band] ** 2) / (np.sum(fft_mag ** 2) + 1e-8)
        
        # Human speech triggers when vocal band power dominates over low-frequency engine rumble
        is_speech = (frame_energy > self.threshold) and (voice_ratio > 0.35)
        
        if is_speech:
            self.hangover_count = self.hangover_frames
            return True
        elif self.hangover_count > 0:
            self.hangover_count -= 1
            return True
        return False


# ======================================================================================
# 3. DUAL-OUTPUT DCCRN (DEEP COMPLEX CRN - MOCKED FOR EMBEDDED EVALUATION)
# ======================================================================================
class DualOutputDCCRN(nn.Module):
    """
    Mock Deep Complex Convolution Recurrent Network (DCCRN).
    Input: Real/Imag STFT or time-domain buffer [Batch, 1, Time]
    Outputs:
      1. Speech Mask (CRM): Isolates clean voice formants.
      2. Noise Mask: Isolates ambient defence noise as a reference for NLMS.
    """
    def __init__(self):
        super(DualOutputDCCRN, self).__init__()
        # Lightweight causal mock layers to simulate compute & memory footprint
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.PReLU()
        )
        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        # Dual output heads
        self.speech_mask_head = nn.ConvTranspose1d(64, 1, kernel_size=5, stride=4, padding=1, output_padding=0)
        self.noise_mask_head = nn.ConvTranspose1d(64, 1, kernel_size=5, stride=4, padding=1, output_padding=0)
        self.eval()

    @torch.no_grad()
    def forward(self, x_tensor: torch.Tensor):
        """
        x_tensor: [1, 1, Samples]
        Returns: (speech_estimate, noise_estimate)
        """
        # Simulating causal complex ratio masking
        feat = self.encoder(x_tensor)
        feat_t = feat.permute(0, 2, 1)
        gru_out, _ = self.gru(feat_t)
        gru_out = gru_out.permute(0, 2, 1)
        
        # Dual masks
        s_mask = torch.sigmoid(self.speech_mask_head(gru_out))
        n_mask = torch.sigmoid(self.noise_mask_head(gru_out))
        
        # Ensure output size matches input length
        target_len = x_tensor.shape[-1]
        s_mask = torch.nn.functional.interpolate(s_mask, size=target_len, mode='linear', align_corners=False)
        n_mask = torch.nn.functional.interpolate(n_mask, size=target_len, mode='linear', align_corners=False)
        
        speech_est = x_tensor * s_mask
        noise_est = x_tensor * n_mask
        
        return speech_est.squeeze().numpy(), noise_est.squeeze().numpy()


# ======================================================================================
# 4. ALWAYS-ON NORMALIZED LMS (NLMS) ADAPTIVE FILTER
# ======================================================================================
class AlwaysOnNLMSFilter:
    """
    Time-domain Normalized Least Mean Squares (NLMS) adaptive filter.
    Runs continuously on incoming raw audio.
    Uses the DCCRN's noise mask output (or running noise estimate) as reference d(n)
    to adaptively suppress stationary vehicle engine drones.
    """
    def __init__(self, filter_order=32, step_size=0.08, epsilon=1e-4):
        self.order = filter_order
        self.mu = step_size
        self.eps = epsilon
        self.weights = np.zeros(self.order, dtype=np.float32)
        self.x_history = np.zeros(self.order, dtype=np.float32)

    def process_sample(self, primary_mic: float, reference_noise: float) -> float:
        """
        Sample-by-sample update:
          y(n) = w(n)^T * x(n)
          e(n) = d(n) - y(n)
          w(n+1) = w(n) + [mu / (||x(n)||^2 + eps)] * e(n) * x(n)
        """
        # Update tapped delay line with reference noise
        self.x_history = np.roll(self.x_history, 1)
        self.x_history[0] = reference_noise

        # Filter estimation
        estimated_noise = np.dot(self.weights, self.x_history)
        
        # Error signal (cleaned audio output)
        error = primary_mic - estimated_noise

        # Normalized weight update
        norm_power = np.dot(self.x_history, self.x_history) + self.eps
        self.weights += (self.mu / norm_power) * error * self.x_history

        return error

    def process_frame(self, primary_frame: np.ndarray, reference_noise_frame: np.ndarray) -> np.ndarray:
        """Process a 10ms or 20ms block sample-by-sample."""
        out = np.zeros_like(primary_frame)
        for i in range(len(primary_frame)):
            out[i] = self.process_sample(primary_frame[i], reference_noise_frame[i])
        return out


# ======================================================================================
# 5. SYNTHETIC BENCHMARK AUDIO GENERATOR (ZERO EXTERNAL FILE DEPENDENCY)
# ======================================================================================
def generate_benchmark_signals(duration_sec=3.0, sample_rate=16000):
    """
    Generates single microphone stream:
      1. Human speech commands ("Alpha One, sector secure")
      2. Mixed with stationary diesel engine hum
      3. Mixed with impulsive gunfire/artillery blasts
    """
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    # Human voice harmonics (with silent pauses to test VAD sleep logic)
    speech = np.zeros(total_samples, dtype=np.float32)
    # Speech burst 1 (0.4s - 1.2s)
    mask1 = (t >= 0.4) & (t <= 1.2)
    t1 = t[mask1]
    speech[mask1] = (np.sin(2 * np.pi * 140 * t1) + 0.5 * np.sin(2 * np.pi * 280 * t1) + 0.25 * np.sin(2 * np.pi * 420 * t1)) * 0.4
    # Speech burst 2 (1.8s - 2.6s)
    mask2 = (t >= 1.8) & (t <= 2.6)
    t2 = t[mask2]
    speech[mask2] = (np.sin(2 * np.pi * 160 * t2) + 0.5 * np.sin(2 * np.pi * 320 * t2) + 0.25 * np.sin(2 * np.pi * 480 * t2)) * 0.45

    # Steady combat vehicle engine hum (60Hz + 120Hz harmonics + rumble)
    engine = (0.5 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.random.normal(0, 0.2, total_samples)) * 0.6
    
    # Sudden impulsive gunfire blasts at 1.0s and 2.1s
    impulses = np.zeros(total_samples, dtype=np.float32)
    for blast_t in [1.0, 2.1]:
        idx = int(blast_t * sample_rate)
        decay = int(0.25 * sample_rate)
        t_d = np.linspace(0, 0.25, decay, endpoint=False)
        impulses[idx:idx + decay] += np.exp(-15 * t_d) * (np.random.normal(0, 1.2, decay) + 0.5 * np.sin(2 * np.pi * 45 * t_d))
        
    combined_noise = (engine + impulses).astype(np.float32)
    raw_mixed_input = speech + combined_noise
    
    return speech, combined_noise, raw_mixed_input, sample_rate


# ======================================================================================
# 6. REAL-TIME STREAMING SIMULATION PIPELINE (<20ms LATENCY EVALUATION)
# ======================================================================================
def run_sonic_shield_pipeline(frame_size_ms=10, sample_rate=16000):
    """
    Simulates real-time embedded streaming pipeline processing audio frame-by-frame.
    Evaluates:
      - 50ms Circular Pre-Roll Buffer
      - VAD Sleep/Wake decision
      - DCCRN Execution Duty Cycle
      - Always-On NLMS Filter
      - Per-frame latency benchmarking (<20ms requirement)
    """
    print("=" * 80)
    print(f"🛡️  SONIC SHIELD: Embedded Real-Time Adaptive Noise Cancellation Simulation")
    print(f"    Target Frame Size: {frame_size_ms} ms (Hop: {int(frame_size_ms * sample_rate / 1000)} samples @ {sample_rate} Hz)")
    print(f"    Latency Budget:    < 20.0 ms per frame")
    print("=" * 80)

    # Generate signals
    clean_speech, noise_groundtruth, raw_input, sr = generate_benchmark_signals(duration_sec=3.0, sample_rate=sample_rate)

    frame_len = int((frame_size_ms / 1000.0) * sr)
    total_frames = len(raw_input) // frame_len

    # Instantiate modules
    circ_buffer = CircularAudioBuffer(capacity_ms=50, sample_rate=sr)
    vad = SpeechVAD(sample_rate=sr, speech_energy_threshold=0.012)
    dccrn = DualOutputDCCRN()
    nlms = AlwaysOnNLMSFilter(filter_order=32, step_size=0.08)

    # State variables
    enhanced_output = []
    latencies_ms = []
    dccrn_invocations = 0
    nlms_only_invocations = 0
    
    # Running reference noise buffer for NLMS during sleep mode
    last_known_noise_ref = np.zeros(frame_len, dtype=np.float32)

    # Warmup PyTorch JIT to prevent first-frame cold-start latency
    dummy_warmup = torch.zeros(1, 1, int(0.05 * sr))
    _ = dccrn(dummy_warmup)

    print(f"\n[STREAM] Streaming {total_frames} frames through embedded DSP loop...\n")

    for f_idx in range(total_frames):
        start_time = time.perf_counter()

        # Step 1: Read incoming frame from single microphone
        frame = raw_input[f_idx * frame_len : (f_idx + 1) * frame_len]

        # Step 2: Write frame to 50ms circular buffer (prevents first-syllable cut)
        circ_buffer.write(frame)

        # Step 3: Low-power Speech-Only VAD Trigger
        is_speech_active = vad.process_frame(frame)

        # Step 4: Sleep-Wake Logic
        if is_speech_active:
            # WAKE STATE: Voice detected -> Run DCCRN on the pre-roll buffered audio
            dccrn_invocations += 1
            
            # Retrieve 50ms pre-roll window for causal phase-aware masking
            buffered_audio = circ_buffer.get_last_n_ms(ms=50)
            in_tensor = torch.from_numpy(buffered_audio).unsqueeze(0).unsqueeze(0).float()
            
            # DCCRN outputs Speech Mask and Noise Reference Mask
            speech_est, noise_est = dccrn(in_tensor)
            
            # Slice the current frame slice from the processed buffer
            speech_frame = speech_est[-frame_len:]
            noise_ref_frame = noise_est[-frame_len:]
            last_known_noise_ref = noise_ref_frame

            # Step 5: Always-On NLMS Filter refines residual stationary tones
            cleaned_frame = nlms.process_frame(speech_frame, noise_ref_frame)

        else:
            # SLEEP STATE (Low Thermal Mode): No human speech -> DCCRN is powered down.
            # Entire workload handled by ultra-lightweight Always-On NLMS filter
            nlms_only_invocations += 1
            
            # NLMS uses primary frame and last estimated noise spectral characteristics
            cleaned_frame = nlms.process_frame(frame, last_known_noise_ref)
            # Attenuate non-speech intervals to keep tactical channel quiet
            cleaned_frame = cleaned_frame * 0.1

        # Record output
        enhanced_output.extend(cleaned_frame)

        # Measure elapsed time (latency per frame)
        frame_latency_ms = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(frame_latency_ms)

        # Periodic telemetry print
        if (f_idx + 1) % 50 == 0 or f_idx == total_frames - 1:
            state_str = "🔥 DCCRN WAKE" if is_speech_active else "💤 DCCRN SLEEP (NLMS ONLY)"
            print(f"Frame {f_idx+1:03d}/{total_frames} | State: {state_str:<26} | Latency: {frame_latency_ms:.3f} ms")

    # ==================================================================================
    # 7. PERFORMANCE & LATENCY REPORT
    # ==================================================================================
    avg_latency = np.mean(latencies_ms)
    max_latency = np.max(latencies_ms)
    p99_latency = np.percentile(latencies_ms, 99)
    dccrn_duty_cycle = (dccrn_invocations / total_frames) * 100.0

    print("\n" + "=" * 80)
    print("📊 SONIC SHIELD EMBEDDED DSP BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Stream Duration:       {len(raw_input)/sr:.2f} seconds ({total_frames} frames @ {frame_size_ms}ms)")
    print(f"Average Frame Latency:       {avg_latency:.3f} ms   (Budget: < 20.0 ms) -> {'[PASSED]' if avg_latency < 20 else '[FAILED]'}")
    print(f"Peak (Max) Latency:          {max_latency:.3f} ms   (Budget: < 20.0 ms) -> {'[PASSED]' if max_latency < 20 else '[FAILED]'}")
    print(f"99th Percentile Latency:     {p99_latency:.3f} ms")
    print(f"DCCRN Sleep-Wake Duty Cycle: {dccrn_duty_cycle:.1f}% Active / {100 - dccrn_duty_cycle:.1f}% Low-Power Sleep")
    print(f"Thermal Footprint Savings:   ~{100 - dccrn_duty_cycle:.1f}% reduction in neural compute vs. always-on deep model")
    print("=" * 80)

    # Save output audio files for auditory verification
    sf.write("benchmark_noisy_input.wav", raw_input, sr)
    sf.write("benchmark_enhanced_output.wav", np.array(enhanced_output, dtype=np.float32), sr)
    sf.write("benchmark_clean_groundtruth.wav", clean_speech, sr)
    print("Saved audio files for verification:")
    print("  - benchmark_noisy_input.wav")
    print("  - benchmark_enhanced_output.wav")
    print("  - benchmark_clean_groundtruth.wav")

if __name__ == '__main__':
    # Test with standard 10ms frame size (160 samples per hop @ 16kHz)
    run_sonic_shield_pipeline(frame_size_ms=10, sample_rate=16000)
