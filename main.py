import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv


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


energy_smooth = np.convolve(energy_norm, np.ones(50) / 50, mode="same")
dt = hop_length / sr

low_thr = 0.25
is_low = energy_smooth < low_thr
min_break_s = 2.0
min_break_n = int(min_break_s / dt)

breaks = []
start = None

for i, low in enumerate(is_low):
    if low and start is None:
        start = i
    elif (not low) and (start is not None):
        end = i
        if (end - start) >= min_break_n:
            breaks.append((start, end))
        start = None

if start is not None:
    end = len(is_low)
    if (end - start) >= min_break_n:
        breaks.append((start, end))

for start, end in breaks:
    t_start = energy_t[start]
    t_end = energy_t[end -1]
    print(f"{t_start:.2f}s -> {t_end:.2f}s")

plt.plot(energy_t, energy_smooth)
plt.xlabel("Temps (s)")
plt.ylabel("Energie normalisée et lissée")

first = True
for start, end in breaks:
    t_start = energy_t[start]
    t_end = energy_t[end -1]
    if first:
        plt.axvspan(t_start, t_end, alpha=0.2, label="Break")
        first = False
    else:
        plt.axvspan(t_start, t_end, alpha=0.2)

plt.legend()
plt.savefig("outputs/breaks.png")
plt.close()


rise_window_s = 0.2
rise_window_n = int(rise_window_s / dt)

rise_thr = 0.10

cooldown_s = 3.0
cooldown_n = int(cooldown_s / dt)

rise = np.zeros_like(energy_smooth)
rise[rise_window_n:] = energy_smooth[rise_window_n:] - energy_smooth[:-rise_window_n]

drops = []
last = -10**9
for i in range(rise_window_n, len(rise)):
    if i - last < cooldown_n:
        continue
    elif rise[i] >= rise_thr:
        drops.append(energy_t[i])
        last = i

plt.plot(energy_t, energy_smooth)
plt.xlabel("Temps (s)")
plt.ylabel("Energie normalisée et lissée")

first = True
for t in drops:
    if first:
        plt.axvline(t, color="red", linestyle="--", alpha=0.8, label="Drop")
        first = False
    else:
        plt.axvline(t, color="red", linestyle="--", alpha=0.8)

plt.legend()
plt.savefig("outputs/drops.png")
plt.close()


with open("outputs/breaks.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["start_s", "end_s"])
    for start, end in breaks:
        t_start = energy_t[start]
        t_end = energy_t[end - 1]
        writer.writerow([t_start, t_end])

with open("outputs/drops.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["drop_s"])
    for t in drops:
        writer.writerow([t])

