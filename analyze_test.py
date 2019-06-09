import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mir_eval
import numpy as np

from hparams import hparams

# test_eval: precision, recall, fscore
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
            id_genre[id_] = line[4]

test_result = dict()  # k: id, v: float
for id_ in id_genre.keys():
    item_truth = np.load(path_test / f'{id_}_truth.npy')
    item_pred = np.load(path_test / f'{id_}_pred.npy')
    test_result[id_] = mir_eval.segment.detection(item_truth, item_pred, trim=True)

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
fig.savefig(path_test / 'test.png', dpi=300)

# genre(precision, recall, f1), total -> CSV
with open(path_test / 'test.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['GENRE',
                     'PRECISION mean', 'PRECISION min', 'PRECISION max',
                     'RECALL mean', 'RECALL min', 'RECALL max',
                     'F1-score mean', 'F1-score min', 'F1-score max'])

    for g, precision, recall, fscore in \
            zip(genre_result.keys(), genre_total[0], genre_total[1], genre_total[2]):
        writer.writerow([g, *precision.tolist(), *recall.tolist(), *fscore.tolist()])

    writer.writerow(['TOTAL', *total_result[0].tolist(), *total_result[1].tolist(), *total_result[2].tolist()])
