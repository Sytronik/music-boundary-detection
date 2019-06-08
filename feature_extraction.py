"""
feature_extraction.py

Usage:
python feature_extraction.py KIND_DATA [--mode=MODE] [--num-workers=N]

- KIND_DATA can be 'train' or 'test'.
- MODE can be 'io', 'in', or 'out' (means what feature will be processed).
    Default is 'io'
- N can be an integer from 1 to cpu_count.
    Default is cpu_count - 1

"""
import csv
import multiprocessing as mp
import os
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import scipy.signal as scsig
from numpy import ndarray
from tqdm import tqdm

from hparams import hparams


def melspectrogram(y: ndarray, sr: int) -> ndarray:
    """

    :param y: (k, n) or (n,). k is no. of audio channels.
    :param sr:
    :return: (k, F, T) or (F, T).
    """
    if y.ndim == 1:
        y = y[np.newaxis, :]  # (1, n)

    S = []
    for item_y in y:
        S.append(
            librosa.stft(item_y,
                         n_fft=hparams.fft_size,
                         hop_length=hparams.hop_size,
                         win_length=hparams.win_size)
        )  # (F[linear], T)

    S = np.stack(S, axis=0)  # (k, F[linear], T)

    if not hasattr(melspectrogram, 'mel_basis'):
        melspectrogram.mel_basis = librosa.filters.mel(sr,
                                                       n_fft=hparams.fft_size,
                                                       n_mels=hparams.num_mels)
    mel_S = np.einsum('mf,kft->kmt', melspectrogram.mel_basis, np.abs(S))  # k, F[mel], T
    logmel_S = np.log10(1 + 10 * mel_S)
    logmel_S = logmel_S.squeeze()  # (1, F, T) -> (F, T)

    return logmel_S


def extract_feature(song_id: int, path_audio: Path, path_annot: Path) \
        -> Optional[Tuple[List[str], List[int], ndarray, ndarray, ndarray, ndarray]]:
    y, _ = librosa.load(str(path_audio), sr=sample_rate, mono=False)  # 2, n
    mel = None
    # print(f'{song_id:4d}: ', end='', flush=True)

    if b_extract_input:
        if kind_data == 'train':
            ys_pitch = {0: y}
            for step, db, F in product(pitchstep, noise_db, max_F_rm):
                # pitch shift
                if step not in ys_pitch:
                    y_temp = [
                        librosa.effects.pitch_shift(one, sample_rate, n_steps=step)
                        for one in y
                    ]
                    ys_pitch[step] = np.stack(y_temp, axis=0)
                y = ys_pitch[step]

                # adding noise
                if db is not None:
                    y += librosa.db_to_amplitude(db) * np.random.randn(*y.shape)

                mel = melspectrogram(y, sample_rate)  # 2, F, T

                # SpecAugmentation
                if F != 0:
                    height = np.random.randint(1, F + 1)
                    f0 = np.random.randint(0, hparams.num_mels - height)
                    mel[..., f0:f0 + height, :] = 0

                np.save(hparams.path_feature[kind_data] / f'{song_id}_{step}_{db}_{F}.npy', mel)
                # print('/', end='', flush=True)
        else:
            mel = melspectrogram(y, sample_rate)
            np.save(hparams.path_feature[kind_data] / f'{song_id}.npy', mel)
            # print('/', end='', flush=True)

    if not b_extract_output:
        return None

    if mel is None:
        y = np.pad(y[0], int(hparams.fft_size // 2), mode='constant')
        mel = librosa.util.frame(y, hparams.win_size, hparams.hop_size)

    # mel = melspectrogram(y, sample_rate)
    n_frames = np.arange(mel.shape[-1] + 1)
    t_frames = n_frames * sample_period * hparams.hop_size

    t_boundaries = []
    sections = []
    with path_annot.open('r', newline='') as f_section:
        annot_sect = [l for l in csv.reader(f_section, delimiter='\t') if float(l[0]) > 0]
        for l_sect in annot_sect:
            t = float(l_sect[0])
            s = l_sect[1]
            # if t == 0:
            #     if s.lower() != 'silence' or c.lower() != 'silence':
            #         raise Exception(song_id)
            #     continue
            if s.lower() == 'end':
                continue
            if sections:
                if sections[-1] == s:
                    continue
                    # raise Exception(song_id)
                # elif sections[-1] == s ^ coarse[-1] == c:
                #     raise Exception(song_id)
            t_boundaries.append(t)
            sections.append(s.lower())

    # s_sect = str(sections).replace('\'', '').replace(',', '')
    # print(f' {s_sect}')

    # boundary label / section segmentation map / coarse structure map
    boundary_label = []
    binary_map = [0]
    sect_map = ['silence']
    i_boundary = 0
    for i_frame in range(len(t_frames) - 1):
        if i_boundary == len(t_boundaries):
            # last section
            boundary_label.append(0)
            sect_map.append(sect_map[-1])
            binary_map.append(binary_map[-1])
        elif t_boundaries[i_boundary] - t_frames[i_frame] > hparams.hop_size * sample_period:
            # if not boundary
            boundary_label.append(0)
            sect_map.append(sect_map[-1])
            binary_map.append(binary_map[-1])
        elif (t_boundaries[i_boundary] - t_frames[i_frame]
              > t_frames[i_frame + 1] - t_boundaries[i_boundary]):
            # if the next frame is closer than the current frame
            boundary_label.append(0)
            sect_map.append(sect_map[-1])
            binary_map.append(binary_map[-1])
        elif (t_boundaries[i_boundary] - t_frames[i_frame]
              <= t_frames[i_frame + 1] - t_boundaries[i_boundary]):
            # if the current frame is closer than the next frame
            if i_frame == 0:  # prevent marking the first frame is the boundary
                boundary_label.append(0)
                sect_map.append(sect_map[-1])
                binary_map.append(binary_map[-1])
            else:
                boundary_label.append(1)
                sect_map.append(sections[i_boundary])
                binary_map.append(int(not binary_map[-1]))
            i_boundary += 1
        elif (t_boundaries[i_boundary] - t_frames[i_frame]
              < t_frames[i_frame - 1] - t_boundaries[i_boundary]):
            # if the current frame is closer than the prev frame
            boundary_label.append(1)
            sect_map.append(sections[i_boundary])
            binary_map.append(int(not binary_map[-1]))
            i_boundary += 1
        else:
            raise Exception(song_id)

    if boundary_label[-1] == 1:
        boundary_label[-1] = 0
        sect_map[-1] = sect_map[-2]
        binary_map[-1] = binary_map[-2]

    sect_map = sect_map[1:]
    boundary_label = np.array(boundary_label)
    binary_map = np.array(binary_map[1:])

    # boundary socre which is gaussian-filtered boundary label
    boundary_score = np.zeros(boundary_label.shape, dtype=np.float32)  # is copied
    boundary_index = np.where(boundary_label == 1)[0]
    for i_boundary in boundary_index:
        i_first = max(i_boundary - half_len_kernel, 0)
        i_last = min(i_boundary + half_len_kernel + 1, len(boundary_score))
        i_k_first = max(+half_len_kernel - i_boundary, 0)
        i_k_last = half_len_kernel + min(half_len_kernel + 1, len(boundary_score) - i_boundary)
        boundary_score[i_first:i_last] += kernel[i_k_first:i_k_last]

    return sections, sect_map, binary_map, boundary_label, boundary_score, boundary_index


def main():
    songs = []
    sect_names = {'silence'}
    sect_maps = dict()
    coarse_maps = dict()
    binary_maps = dict()
    boundary_labels = dict()
    boundary_scores = dict()
    boundary_indexes = dict()

    pool = mp.Pool(num_workers)
    # pool = mp.Pool(4)
    results = dict()
    with path_metadata.open('r', newline='') as f_metadata:
        pbar_meta = tqdm(csv.reader(f_metadata), dynamic_ncols=True)
        for idx, l_meta in enumerate(pbar_meta):
            if idx == 0:
                idx_discard_flag = l_meta.index('SONG_WAS_DISCARDED_FLAG')
                continue
            song_id = int(l_meta[0])

            path_audio = path_audio_dir / f'{song_id}.mp3'
            path_annot_1 = path_annot_dir / f'{song_id}/parsed/textfile1_functions.txt'
            path_annot_2 = path_annot_dir / f'{song_id}/parsed/textfile2_functions.txt'
            if path_annot_1.exists():
                path_annot = path_annot_1
            elif path_annot_2.exists():
                path_annot = path_annot_2
            else:
                path_annot = None
            if (not path_audio.exists() or l_meta[idx_discard_flag] == 'TRUE'
                    or not path_annot):
                continue
            songs.append(str(song_id))
            pbar_meta.set_postfix_str(f'[{len(songs)} songs]')
            results[song_id] = pool.apply_async(extract_feature,
                                                (song_id, path_audio, path_annot))
            # results[song_id] = extract_feature(song_id, path_audio, path_annot)

    pbar = tqdm(results.items(), dynamic_ncols=True)
    for song_id, result in pbar:
        if b_extract_output:
            s_song_id = str(song_id)
            (
                sections,
                sect_map, binary_map,
                boundary_label, boundary_score, boundary_index
            ) = result.get()

            sect_names = sect_names.union(sections)
            boundary_labels[s_song_id] = boundary_label
            sect_maps[s_song_id] = sect_map
            binary_maps[s_song_id] = binary_map
            boundary_scores[s_song_id] = boundary_score
            boundary_indexes[s_song_id] = boundary_index

            pbar.write(str(sections).replace('\'', '').replace(',', ''))
        else:
            result.get()

    if b_extract_output:
        print()
        print('save ground truth...')

        if kind_data == 'test':
            sect_names_train = []
            with (hparams.path_feature['train'] / 'section_names.txt').open('r') as f:
                for line in f.readlines():
                    sect_names_train.append(line.replace('\n', '').split(': ')[1].lower())

            if any([name not in sect_names_train for name in sect_names]):
                raise Exception
            sect_names = sect_names_train

        dict_sect_idx = {name: idx for idx, name in enumerate(sect_names)}
        for s_song_id in sect_maps.keys():
            sect_maps[s_song_id] = np.array([dict_sect_idx[s] for s in sect_maps[s_song_id]])

            coarse_map = []
            dict_coarse_idx = {}
            idx = 0
            for c in sect_maps[s_song_id]:
                if c not in dict_coarse_idx:
                    dict_coarse_idx[c] = idx
                    idx += 1
                coarse_map.append(dict_coarse_idx[c])
            coarse_maps[s_song_id] = np.array(coarse_map)

        np.savez(path_feature / 'section_maps.npz', **sect_maps)
        np.savez(path_feature / 'coarse_maps.npz', **coarse_maps)
        np.savez(path_feature / 'binary_maps.npz', **binary_maps)
        np.savez(path_feature / 'boundary_labels.npz', **boundary_labels)
        np.savez(path_feature / f'boundary_scores_{len_kernel}.npz', **boundary_scores)
        np.savez(path_feature / 'boundary_indexes.npz', **boundary_indexes)

        with (path_feature / 'section_names.txt').open('w') as f:
            for idx, name in enumerate(sect_names):
                f.write(f'{idx:3d}: {name}\n')

        with (hparams.path_feature[kind_data] / 'songs.txt').open('w') as f:
            f.writelines(songs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('kind_data', choices=('train', 'test'))
    parser.add_argument('--mode', choices=('in', 'out', 'io'), default='io')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1)
    args = parser.parse_args()

    kind_data = args.kind_data
    if not hparams.path_feature[kind_data].exists():
        os.makedirs(hparams.path_feature[kind_data])

    b_extract_input = False if args.mode == 'out' else True
    b_extract_output = False if args.mode == 'in' else True

    num_workers = args.num_workers

    path_metadata = hparams.path_dataset[kind_data] / 'metadata/metadata.csv'
    path_audio_dir = hparams.path_dataset[kind_data] / 'audio'
    path_annot_dir = hparams.path_dataset[kind_data] / 'annotations'
    path_feature = hparams.path_feature[kind_data]

    pitchstep = hparams.pitchstep
    noise_db = hparams.noise_db
    max_F_rm = hparams.max_F_rm

    sample_rate = hparams.sample_rate
    sample_period = 1 / sample_rate

    len_kernel = hparams.len_gaussian_kernel
    sigma = len_kernel / 4
    half_len_kernel = len_kernel // 2
    kernel = scsig.gaussian(hparams.len_gaussian_kernel, sigma)

    main()
