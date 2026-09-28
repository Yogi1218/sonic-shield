import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa

class DefenceNoiseDataset(Dataset):
    def __init__(self, speech_dir=None, noise_dir=None, sample_rate=16000, duration=2.0, num_samples=1000, is_train=True):
        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = num_samples
        self.is_train = is_train
        self.length = int(self.sample_rate * self.duration)
        
        self.speech_files = []
        if speech_dir and os.path.exists(speech_dir):
            for ext in ('*.wav', '*.flac', '*.mp3'):
                self.speech_files.extend(glob.glob(os.path.join(speech_dir, '**', ext), recursive=True))
        
        self.noise_files = []
        if noise_dir and os.path.exists(noise_dir):
            for ext in ('*.wav', '*.flac', '*.mp3'):
                self.noise_files.extend(glob.glob(os.path.join(noise_dir, '**', ext), recursive=True))
                
        self.noise_types = ['gunshot', 'helicopter', 'siren', 'artillery', 'stationary_hum']

    def __len__(self):
        return self.num_samples

    def _generate_synthetic_speech(self):
        t = np.linspace(0, self.duration, self.length, endpoint=False)
        f0 = 120 + 30 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(f0) / self.sample_rate
        speech = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.125 * np.sin(4 * phase)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3 * t)) * (0.5 * (1.0 + np.sin(2 * np.pi * 0.5 * t)))
        speech = speech * envelope
        fricative_env = (1.0 - envelope) * (np.random.random(self.length) > 0.95)
        speech += np.random.normal(0, 0.05, self.length) * fricative_env
        speech = speech / (np.max(np.abs(speech)) + 1e-8)
        return speech.astype(np.float32)

    def _generate_synthetic_noise(self, noise_type):
        t = np.linspace(0, self.duration, self.length, endpoint=False)
        noise = np.zeros(self.length, dtype=np.float32)
        
        if noise_type == 'gunshot':
            num_shots = random.randint(1, 3)
            for _ in range(num_shots):
                onset = random.randint(0, int(self.length * 0.8))
                decay = random.uniform(10, 30)
                env = np.zeros(self.length)
                duration_samples = self.length - onset
                t_decay = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)
                env[onset:] = np.exp(-decay * t_decay)
                shot_noise = np.random.normal(0, 0.8, self.length) + 0.5 * np.sin(2 * np.pi * 80 * t)
                noise += (shot_noise * env).astype(np.float32)
                
        elif noise_type == 'helicopter':
            blade_rate = random.uniform(8.0, 15.0)
            pulses = 0.5 * (1.0 + np.sin(2 * np.pi * blade_rate * t)) ** 4
            carrier = 0.4 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 120 * t)
            rotor_thud = pulses * carrier
            wind = np.random.normal(0, 0.15, self.length)
            wind_mod = (0.7 + 0.3 * pulses) * wind
            noise = (rotor_thud + wind_mod).astype(np.float32)
            
        elif noise_type == 'siren':
            sweep_freq = 0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * t)
            f_inst = 400 + 400 * sweep_freq
            phase = 2 * np.pi * np.cumsum(f_inst) / self.sample_rate
            noise = np.sin(phase).astype(np.float32)
            
        elif noise_type == 'artillery':
            onset = random.randint(0, int(self.length * 0.5))
            decay = random.uniform(3, 8)
            env = np.zeros(self.length)
            duration_samples = self.length - onset
            t_decay = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)
            env[onset:] = np.exp(-decay * t_decay)
            rumble = np.sin(2 * np.pi * 50 * t) * 0.6 + np.sin(2 * np.pi * 100 * t) * 0.3
            noise = (rumble * env + np.random.normal(0, 0.05, self.length) * env).astype(np.float32)
            
        elif noise_type == 'stationary_hum':
            engine = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)
            bg_noise = np.random.normal(0, 0.2, self.length)
            noise = (0.6 * engine + 0.4 * bg_noise).astype(np.float32)
            
        noise = noise / (np.max(np.abs(noise)) + 1e-8)
        return noise.astype(np.float32)
