import csv
from argparse import ArgumentParser
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Set, Optional, Sequence

import librosa.display
import matplotlib.ticker as tckr
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
from numpy import ndarray
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hparams import hparams


def draw_mel_boundary(path_audio: Path, path_figure: Path,
                      score_out: ndarray, prediction: ndarray, truth: ndarray,
                      threshold: float,
                      draw_title=False,
                      draw_legend=True,
                      xlim: Optional[Sequence[float]] = None,
                      xticklabels: Optional[Sequence[float]] = None,
                      ):
    audio, _ = librosa.core.load(str(path_audio), sr=hparams.sample_rate)
    mel_S = librosa.feature.melspectrogram(audio,
                                           sr=hparams.sample_rate,
                                           n_fft=hparams.fft_size,
                                           hop_length=hparams.hop_size,
                                           n_mels=hparams.num_mels)

    t_axis = np.arange(len(score_out)) * hparams.hop_size / hparams.sample_rate
    # figure
    if xlim is not None:
        duration = xlim[1] - xlim[0]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(duration / 20 + 1.5, 5))
        # ax for colorbar
        ax_cbar = None
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        # ax for colorbar
        ax_cbar = make_axes_locatable(ax1).append_axes('right', size=0.1, pad=0.05)
        ax_none = make_axes_locatable(ax2).append_axes('right', size=0.1, pad=0.05)
        ax_none.set_visible(False)

    c_vline_pred = 'C2'
    c_vline_truth = 'C9'

    # ax1: mel spectrogram
    librosa.display.specshow(librosa.power_to_db(mel_S, ref=np.max),
                             x_axis='time', y_axis='mel',
                             sr=hparams.sample_rate,
                             hop_length=hparams.hop_size,
                             ax=ax1,
                             )
    # prediction and target boundary
    ax1.vlines(x=prediction,
               ymin=4000, ymax=16000,
               colors=c_vline_pred, label='prediction', zorder=2)
    ax1.vlines(x=truth,
               ymin=0, ymax=600,
               colors=c_vline_truth, label='truth', zorder=2)
    if ax_cbar:
        fig.colorbar(ax1.collections[0], format='%+2.0f dB', cax=ax_cbar)
    if draw_title:
        ax1.set_title('mel spectrogram')
    x_formatter = ax1.xaxis.get_major_formatter()
    ax1.xaxis.set_major_locator(tckr.MultipleLocator(30))
    ax1.xaxis.set_minor_locator(tckr.MultipleLocator(10))
    ax1.set_xlabel('time (min:sec)')

    # ax2: boundary score
    ax2.plot(t_axis, score_out,
             color='C1', zorder=1, label='estimated boundary score',
             linewidth=0.75)
    # prediction and target boundary
    ylim = [-0.3, 1.3]
    ax2.vlines(x=prediction,
               ymin=0.9, ymax=ylim[1],
               colors=c_vline_pred, label='predicted boundary', zorder=2)
    ax2.vlines(x=truth,
               ymin=ylim[0], ymax=0.1,
               colors=c_vline_truth, label='target boundary', zorder=2)
    if draw_legend:
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)

    ax2.set_xlim(ax1.get_xlim())
    ax2.xaxis.set_major_formatter(x_formatter)
    ax2.xaxis.set_major_locator(ax1.xaxis.get_major_locator())
    ax2.xaxis.set_minor_locator(ax1.xaxis.get_minor_locator())
    ax2.set_xlabel('time (min:sec)')

    ax2.set_ylim(*ylim)
    ax2.set_yticks([0, 1])
    ax2.set_yticks([threshold], minor=True)
    ax2.set_yticklabels(['threshold'], minor=True)

    ax2.grid(True, which='major', axis='y')
    ax2.grid(True, which='minor', axis='y', linestyle='--', linewidth=1)

    if xlim is not None:
        ax1.set_xlim(*xlim)
        ax2.set_xlim(*xlim)
        ax1.set_xticks(xlim)
        ax2.set_xticks(xlim)
        ax1.set_xticklabels(xticklabels)
        ax2.set_xticklabels(xticklabels)

    fig.tight_layout()
    fig.savefig(path_figure, dpi=600)


def main(test_epoch: int, ids_drawn: Set[int], tol: float):
    """

    :param test_epoch:
    :param ids_drawn: song ids to be plotted in mel and boundary.
    :param tol: hit rate tolerance
    :return:
    """
    # test_eval: precision, recall, fscore
    path_test = Path(hparams.logdir, f'test_{test_epoch}')
    if not path_test.exists():
        raise FileNotFoundError(path_test)
    path_metadata = hparams.path_dataset['test'] / 'metadata/metadata.csv'

    # Take the genres of each song in id order
    ids = []
    id_genre = []  # k: id, v: genre
    i_col_genre = 3
    with path_metadata.open('r', encoding='utf-8') as f:
        read = csv.reader(f)
        for idx, line in enumerate(read):
            if idx == 0:
                i_col_genre = line.index('GENRE')
                continue
            id_ = line[0]
            if (path_test / f'{id_}_pred.npy').exists():
                ids.append(int(id_))
                id_genre.append(line[i_col_genre])

    # measure
    all_results = []  # k: id, v: float(precision, recall, F1, F0.58)
    for i_id, id_ in enumerate(ids):
        item_truth = np.load(path_test / f'{id_}_truth.npy')
        item_pred = np.load(path_test / f'{id_}_pred.npy')
        prec, recall, f1 = mir_eval.segment.detection(item_truth, item_pred, trim=True, window=tol)
        _, _, f058 = mir_eval.segment.detection(item_truth, item_pred, beta=0.58, trim=True,
                                                window=tol)
        all_results.append(np.array((prec, recall, f1, f058)))

    # total mean / min / max
    all_results = np.stack(all_results, axis=0)  # (N, 4)
    total_mean = np.mean(all_results, axis=0)  # (4,)
    total_min = np.min(all_results, axis=0)  # (4,)
    total_max = np.max(all_results, axis=0)  # (4,)
    ids_drawn.add(int(ids[np.argmin(all_results, axis=0)[2]]))
    ids_drawn.add(int(ids[np.argmax(all_results, axis=0)[2]]))
    total_min_err = total_mean - total_min
    total_max_err = total_max - total_mean
    total_errs = np.stack((total_min_err, total_max_err), axis=0)  # 2, 4
    total_stacked = np.stack((total_mean, total_min, total_max), axis=-1)  # (4, 3)

    # mean / min / max per genres
    genre_result = defaultdict(list)  # k: genre, v: list
    for i_id, g in enumerate(id_genre):
        genre_result[g].append(all_results[i_id])

    all_genres = list(genre_result.keys())
    num_genres = len(genre_result)
    xs = np.arange(num_genres + 1)
    genre_mean = np.zeros((num_genres, 4))
    genre_max = np.zeros((num_genres, 4))
    genre_min = np.zeros((num_genres, 4))
    for idx, g in enumerate(all_genres):
        genre_mean[idx] = np.mean(genre_result[g], axis=0)
        genre_max[idx] = np.max(genre_result[g], axis=0)
        genre_min[idx] = np.min(genre_result[g], axis=0)

    genre_min_err = genre_mean - genre_min
    genre_max_err = genre_max - genre_mean
    genre_errs = np.stack((genre_min_err, genre_max_err), axis=0)  # 2, num_genres, 4

    genre_stacked = np.stack((genre_mean.T, genre_min.T, genre_max.T), axis=-1)  # 4, genre, 3

    # figure
    fig, ax = plt.subplots()
    common_ebar_kwargs = dict(elinewidth=0.75, capsize=3, linestyle='', marker='o')
    ax.errorbar(xs[:-1], genre_mean[:, 2], yerr=genre_errs[:, :, 2],
                **common_ebar_kwargs)
    ax.errorbar(xs[-1], total_mean[2], yerr=total_errs[:, 2:3],
                color='black',
                **common_ebar_kwargs)
    for x, y in zip(xs[:-1], genre_mean[:, 2]):
        ax.text(x + 0.1, y, f'{y:.3f}')
    ax.text(xs[-1] + 0.1, total_mean[2], f'{total_mean[2]:.3f}')

    ax.set_xticks(xs)
    ax.set_xticklabels([*all_genres, 'Total'], rotation='vertical')
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.9)
    # ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylim(0, 1)

    ax.set_ylabel('F1 Score')
    ax.grid(True, axis='y')

    fig.tight_layout()
    fig.savefig(path_test / 'test_genre.png', dpi=300)

    # genre(precision, recall, F1, F0.58), total -> CSV
    with open(path_test / 'test.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(
            ['GENRE',
             *list(product(('PRECISION', 'RECALL', 'F1', 'F0.58'),
                           ('mean', 'min', 'max'))),
             ]
        )
        for idx, g in enumerate(all_genres):
            writer.writerow([g, *genre_stacked[:, idx, :].flatten().tolist()])

        writer.writerow(['TOTAL', *total_stacked.flatten().tolist()])

    # Draw mel-spectrogram and boundary detection result
    try:
        thresholds = dict(**np.load(path_test / 'thresholds.npz'))
    except IOError:
        thresholds = None
    for id_ in ids_drawn:
        score_out = np.load(path_test / f'{id_}.npy')
        prediction = np.load(path_test / f'{id_}_pred.npy')[:, 0]
        truth = np.load(path_test / f'{id_}_truth.npy')[:, 0]
        draw_mel_boundary(hparams.path_dataset['test'] / f'audio/{id_}.mp3',
                          path_test / f'test_boundary_{id_}.png',
                          score_out,
                          prediction,
                          truth,
                          thresholds[str(id_)] if thresholds else 0.5,
                          # draw_legend=False if id_ == 18 else True,
                          # xlim=(130, 140) if id_ == 18 else None,
                          # xticklabels=('2:10', '2:20') if id_ == 18 else None,
                          )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('epoch', type=int)
    parser.add_argument('--song', default='set()')
    parser.add_argument('--tol', default=0.5)

    args = hparams.parse_argument(parser, print_argument=False)
    plt.rc('font', family='Arial', size=12)
    s_songs = eval(args.song)
    assert isinstance(s_songs, set)
    main(args.epoch, s_songs, args.tol)
