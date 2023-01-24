import librosa 
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

if __name__ == "__main__":
    filename = "./a.mp3" # librosa.ex('trumpet')
    y_stereo, sr = librosa.load(filename, mono=False)
    y = y_stereo[1]

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512,
                                    fmax=20000)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()