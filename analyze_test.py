import csv
from collections import defaultdict
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import mir_eval
import numpy as np

from hparams import hparams

path_test = Path(hparams.logdir, 'test')
path_metadata = hparams.path_dataset['test'] / 'metadata/metadata.csv'

# Take the genres of each song in id order
with path_metadata.open('r', encoding='utf-8') as f:
    read = csv.reader(f)
    id_genre = dict()  # k: id, v: genre
    for idx, line in enumerate(read):
        if idx == 0:
            continue
        id_ = line[0]
        if (path_test / f'{id_}_pred.npy').exists():
            id_genre[id_] = line[3]

test_result = dict()  # k: id, v: float (precision, recall, F1, F0.58)
for id_ in id_genre.keys():
    item_truth = np.load(path_test / f'{id_}_truth.npy')
    item_pred = np.load(path_test / f'{id_}_pred.npy')
    result = mir_eval.segment.detection(item_truth, item_pred, trim=True)
    _, _, result_ = mir_eval.segment.detection(item_truth, item_pred, beta=0.58, trim=True)
    test_result[id_] = [*result, result_]

total = list(test_result.values())
total_mean = np.mean(total, axis=0)
total_max = np.max(total, axis=0)
total_min = np.min(total, axis=0)
total_result = np.array([total_mean, total_min, total_max])
total_result = total_result.transpose((1, 0))

genre_result = defaultdict(list)  # k: genre, v: list
for id_, g in id_genre.items():
    genre_result[g].append(test_result[id_])

all_genre = list(genre_result.values())
genre_mean = []
genre_max = []
genre_min = []
for list_result in genre_result.values():
    genre_mean.append(np.mean(list_result, axis=0))
    genre_max.append(np.max(list_result, axis=0))
    genre_min.append(np.min(list_result, axis=0))

genre_mean = np.array(genre_mean)
min_err = genre_mean - np.array(genre_min)
max_err = np.array(genre_max) - genre_mean
genre_err = np.array([min_err, max_err])
genre_total = np.array([genre_mean, genre_min, genre_max])
genre_total = genre_total.transpose((2, 1, 0))

xs = np.arange(len(genre_result))
plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots()
ax.errorbar(xs, genre_mean[:, 2], yerr=genre_err[:, :, 2],
            linestyle='', marker='o')
ax.set_xticks(xs)
ax.set_xticklabels(genre_result)
ax.set_ylabel('F1-score')
for x, y in zip(xs, genre_mean[:, 2]):
    plt.text(x + 0.1, y, f'{y:.3f}')
ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
fig.savefig(path_test / 'test_genre.png', dpi=300)

# genre(precision, recall, F1, F0.58), total -> CSV
with open(path_test / 'test.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['GENRE',
                     'PRECISION mean', 'PRECISION min', 'PRECISION max',
                     'RECALL mean', 'RECALL min', 'RECALL max',
                     'F1-score mean', 'F1-score min', 'F1-score max',
                     'F0.58-score mean', 'F0.58-score min', 'F0.58-score max'])

    for g, precision, recall, fone, fzero in \
            zip(genre_result.keys(), genre_total[0], genre_total[1], genre_total[2], genre_total[3]):
        writer.writerow([g, *precision.tolist(), *recall.tolist(), *fone.tolist(), *fzero.tolist()])

    writer.writerow(['TOTAL', *total_result[0].tolist(), *total_result[1].tolist(),
                     *total_result[2].tolist(), *total_result[3]])

# Draw mel-spectrogram and boundary detection result
s_id = 12
song, _ = librosa.core.load(hparams.path_dataset['test'] / f'audio/{s_id}.mp3', sr=hparams.sample_rate)
mel_S = librosa.feature.melspectrogram(song, hparams.sample_rate, n_fft=hparams.fft_size,
                                       hop_length=hparams.hop_size, n_mels=hparams.num_mels)

curve = np.load(path_test / f'{s_id}.npy')[:mel_S.shape[1]]
prediction = np.load(path_test / f'{s_id}_pred.npy')[:, 0]
truth = np.load(path_test / f'{s_id}_truth.npy')[:, 0]
t_axis = np.arange(len(curve)) * hparams.hop_size / hparams.sample_rate

fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.power_to_db(mel_S, ref=np.max),
                         x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
ax1.vlines(x=prediction, ymin=4000, ymax=16000, colors='r', label='prediction', zorder=2)
ax1.vlines(x=truth, ymin=0, ymax=600, colors='w', label='truth', zorder=2)
ax1.set_title(str(s_id))
ax1.legend(loc='upper right')
ax1.set_xlabel('Time')

ax2 = plt.subplot(2, 1, 2)
plt.plot(t_axis, curve, zorder=1)
ax2.vlines(x=prediction, ymin=0.7, ymax=1, colors='r', label='prediction', zorder=2)
ax2.vlines(x=truth, ymin=0, ymax=0.3, colors='y', label='truth', zorder=2)
ax2.set_title(str(s_id))
ax2.legend(loc='upper right')
ax2.set_yticks([0, 0.5, 1])
ax2.grid(True, axis='y')
ax2.set_xlabel('time (sec)')
ax2.set_ylabel('boundary score')

fig.tight_layout()
fig.savefig(path_test / 'test_boundary.png', dpi=300)
