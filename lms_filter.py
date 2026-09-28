import numpy as np

class NormalizedLMSFilter:
    def __init__(self, num_taps=32, step_size=0.08, eps=1e-4):
        self.num_taps = num_taps
        self.step_size = step_size
        self.eps = eps
        
    def filter(self, dccrn_signal, noisy_signal):
        N = len(dccrn_signal)
        noise_ref = noisy_signal - dccrn_signal
        w = np.zeros(self.num_taps)
        buffer = np.zeros(self.num_taps)
        cleaned_signal = np.zeros(N)
        
        for n in range(N):
            buffer = np.roll(buffer, 1)
            buffer[0] = noise_ref[n]
            y_hat = np.dot(w, buffer)
            e = dccrn_signal[n] - y_hat
            cleaned_signal[n] = e
            norm = np.dot(buffer, buffer) + self.eps
            w += (self.step_size / norm) * e * buffer
            
        return cleaned_signal
