import numpy as np
import matplotlib.pyplot as plt
from hparams import hparams

out = np.random.exponential(scale=1, size=10000)
plt_out = plt.plot(out)
plt.show()

N_6s = round(6 * hparams.sample_rate / hparams.hop_size) - 1
N_12s = round(12 * hparams.sample_rate / hparams.hop_size) -1

candid_val = []
candid_idx = []
for i in range(len(out)):
    if i < N_6s:
        if out[i] >= np.amax(out[:i + N_6s + 1]):
            candid_val.append(out[i])
            candid_idx.append(i)
    else:
        if out[i] >= np.amax(out[i - N_6s:i + N_6s + 1]):
            candid_val.append(out[i])
            candid_idx.append(i)

threshold = []
for j in candid_idx:
    if j < N_12s:
        threshold.append(out[j] - np.mean(out[0:j + N_6s + 1]))
    else:
        threshold.append(out[j] - np.mean(out[j - N_12s:j + N_6s + 1]))

boundary_idx = np.where(out > np.amax(threshold))