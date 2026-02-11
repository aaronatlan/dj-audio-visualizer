import librosa
import numpy as np
import matplotlib.pyplot as plt


y, sr = librosa.load("audio/track.flac")
t = np.arange(len(y)) / sr

plt.plot(t, y)
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Waveform du morceau")
plt.savefig("outputs/waveform.png")
plt.close()


frame_length = 2048
hop_length = 512

frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
energy = np.sum(frames ** 2, axis = 0)
energy_t = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
energy_max = np.max(energy)
energy_norm = energy / energy_max

plt.plot(energy_t, energy_norm)
plt.xlabel("Temps (s)")
plt.ylabel("Energie normalisée")
plt.title("Energie du morceau")
plt.savefig("outputs/energy.png")
plt.close()