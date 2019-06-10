import csv
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mir_eval
import numpy as np

from hparams import hparams


def main():
    # test_eval: precision, recall, fscore
    path_test = Path(hparams.logdir, f'test_{test_epoch}')
    path_metadata = hparams.path_dataset['test'] / 'metadata/metadata.csv'

# Take the genres of each song in id order
    with path_metadata.open('r', encoding='utf-8') as f:
        read = csv.reader(f)
        id_genre = dict()  # k: id, v: genre
        i_col_genre = 3
        for idx, line in enumerate(read):
            if idx == 0:
                i_col_genre = line.index('GENRE')
                continue
            id_ = line[0]
            if (path_test / f'{id_}_pred.npy').exists():
                id_genre[id_] = line[i_col_genre]

    test_result = dict()  # k: id, v: float
    for id_ in id_genre.keys():
        item_truth = np.load(path_test / f'{id_}_truth.npy')
        item_pred = np.load(path_test / f'{id_}_pred.npy')
        test_result[id_] = mir_eval.segment.detection(item_truth, item_pred, trim=True)

    total = list(test_result.values())  # (N, 3) list
    total_mean = np.mean(total, axis=0)  # (3,)
    total_min = np.min(total, axis=0)  # (3,)
    total_max = np.max(total, axis=0)  # (3,)
    total_result = np.array([total_mean, total_min, total_max])  # (3, 3)
    total_result = total_result.T  # (3: metrics, 3: mean-min-max)

    genre_result = defaultdict(list)  # k: genre, v: list
    for id_, g in id_genre.items():
        genre_result[g].append(test_result[id_])

    all_genre = list(genre_result.keys())
    num_genres = len(genre_result)
    genre_mean = np.zeros((num_genres, 3))
    genre_max = np.zeros((num_genres, 3))
    genre_min = np.zeros((num_genres, 3))
    for idx, g in enumerate(all_genre):
        genre_mean[idx] = np.mean(genre_result[g], axis=0)
        genre_max[idx] = np.max(genre_result[g], axis=0)
        genre_min[idx] = np.min(genre_result[g], axis=0)

    # genre_mean = np.array(genre_mean)
    min_err = genre_mean - genre_min
    max_err = genre_max - genre_mean
    genre_err = np.stack((min_err, max_err), axis=0)  # 2, num_genres, 3
    genre_total = np.array([genre_mean, genre_min, genre_max])  # 3, genre, 3
    genre_total = genre_total.transpose((2, 1, 0))  # 3: metrics, num_genres, 3: mean-min-max

    fig, ax = plt.subplots()
    ax.errorbar(np.arange(len(genre_result)), genre_mean[:, 2], yerr=genre_err[:, :, 2],
                linestyle='', marker='o')
    ax.set_xticklabels(genre_result)
    plt.show()

    # genre(precision, recall, f1), total -> CSV
    with open(path_test / 'test.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['GENRE',
                         'PRECISION mean', 'PRECISION min', 'PRECISION max',
                         'RECALL mean', 'RECALL min', 'RECALL max',
                         'F1-score mean', 'F1-score min', 'F1-score max'])

        for idx, g in enumerate(all_genre):
            writer.writerow([g, *genre_total[:, idx, :].flatten().tolist()])

        writer.writerow(['TOTAL', *total_result.flatten().tolist()])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('epoch', type=int)

    args = hparams.parse_argument(parser)
    test_epoch = args.epoch
    main()
