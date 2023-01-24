import requests

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fftshift
import matplotlib.pyplot as plt
rng = np.random.default_rng()

if __name__ == "__main__":

    audio_file = "./a.wav"

    sample_rate, samples = wavfile.read(audio_file)

    # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    f, t, Sxx = signal.spectrogram(samples, sample_rate, return_onesided=False)
    print(f.shape, t.shape, Sxx.shape)
    f = f[:100]
    t = t[:100]
    Sxx = Sxx[:100][:100]
    plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # plt.pcolormesh(times, frequencies, spectrogram, shading='gouraud', vmin=spectrogram.min(), vmax=spectrogram.max())
    # plt.imshow(spectrogram)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # # plt.colorbar()
    # plt.show()