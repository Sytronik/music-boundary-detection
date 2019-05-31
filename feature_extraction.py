"""
feature_extraction.py

A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
"""
import csv
import os
from argparse import ArgumentParser
from itertools import product
import multiprocessing as mp
from pathlib import Path

import librosa
import numpy as np
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
        y = y.unsqueeze(0)  # (1, n)

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


def extract_feature(song_id: int, path_audio: Path, path_annot: Path):
    y, _ = librosa.load(str(path_audio), sr=sample_rate, mono=False)  # 2, n
    # length = y.shape[1]
    # print(f'{song_id:4d}: ', end='', flush=True)

    if kind_data == 'train':
        ys_pitch = {0: y}
        for step, db, F in product(pitchstep, noise_db, max_F_rm):
            # pitch shift
            if step not in ys_pitch:
                y_temp = [librosa.effects.pitch_shift(one, sample_rate, n_steps=step) for one in y]
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

    n_frames = np.arange(mel.shape[-1] + 1)
    t_frames = n_frames * T * hparams.hop_size

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
            sections.append(s)

    # s_sect = str(sections).replace('\'', '').replace(',', '')
    # print(f' {s_sect}')

    # boundary label / section segmentation map / coarse structure map
    boundary_labels = []
    sect_map = ['Silence']
    i_boundary = 0
    for i_frame in range(len(t_frames) - 1):
        if i_boundary == len(t_boundaries):
            # last section
            boundary_labels.append(0)
            sect_map.append(sect_map[-1])
        elif t_boundaries[i_boundary] - t_frames[i_frame] > hparams.hop_size * T:
            # if not boundary
            boundary_labels.append(0)
            sect_map.append(sect_map[-1])
        elif (t_boundaries[i_boundary] - t_frames[i_frame]
              > t_frames[i_frame + 1] - t_boundaries[i_boundary]):
            # if the next frame is closer than the current frame
            boundary_labels.append(0)
            sect_map.append(sect_map[-1])
        elif (t_boundaries[i_boundary] - t_frames[i_frame]
              <= t_frames[i_frame + 1] - t_boundaries[i_boundary]):
            # if the current frame is closer than the next frame
            boundary_labels.append(1)
            sect_map.append(sections[i_boundary])
            i_boundary += 1
        elif (t_boundaries[i_boundary] - t_frames[i_frame]
              < t_frames[i_frame - 1] - t_boundaries[i_boundary]):
            # if the current frame is closer than the prev frame
            boundary_labels.append(1)
            sect_map.append(sections[i_boundary])
            i_boundary += 1
        else:
            raise Exception(song_id)

    return sections, sect_map


def main():
    songs = []
    sect_names = {'Silence'}
    sect_maps = dict()

    pool = mp.Pool(mp.cpu_count()-1)
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
            songs.append(song_id)
            pbar_meta.set_postfix_str(f'[{len(songs)} songs]')
            results[song_id] = pool.apply_async(extract_feature,
                                                (song_id, path_audio, path_annot))
            # results[song_id] = extract_feature(song_id, path_audio, path_annot)

    pbar = tqdm(results.items(), dynamic_ncols=True)
    for song_id, result in pbar:
        sections, sect_map = result.get()
        # sections, sect_map = result
        sect_names.union(sections)
        sect_maps[song_id] = sect_map
        pbar.write(str(sections).replace('\'', '').replace(',', ''))

    print()
    print('save maps...')

    dict_sect_idx = {name: idx for idx, name in enumerate(sect_names)}
    coarse_maps = dict()
    for song_id in sect_maps.keys():
        coarse_map = []
        dict_coarse_idx = {}
        idx = 0
        for c in sect_maps[song_id]:
            if c not in dict_coarse_idx:
                dict_coarse_idx[c] = idx
                idx += 1
            coarse_map.append(dict_coarse_idx[c])
        coarse_maps[song_id] = np.array(coarse_map)
        sect_maps[song_id] = np.array([dict_sect_idx[s] for s in sect_maps[song_id]])

    np.savez(path_feature / 'section_maps.npz', **sect_maps)
    np.savez(path_feature / 'coarse_maps.npz', **coarse_maps)
    with (path_feature / 'section_names.txt').open('w') as f:
        for idx, name in enumerate(sect_names):
            f.write(f'{idx:3d}: {name}\n')
    with (hparams.path_feature[kind_data] / 'songs.txt').open('w') as f:
        f.writelines(songs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('kind_data', choices=('train', 'test'))
    args = parser.parse_args()
    kind_data = args.kind_data
    if not hparams.path_feature[kind_data].exists():
        os.makedirs(hparams.path_feature[kind_data])

    path_metadata = hparams.path_dataset[kind_data] / 'metadata/metadata.csv'
    path_audio_dir = hparams.path_dataset[kind_data] / 'audio'
    path_annot_dir = hparams.path_dataset[kind_data] / 'annotations'
    path_feature = hparams.path_feature[kind_data]

    pitchstep = hparams.pitchstep
    noise_db = hparams.noise_db
    max_F_rm = hparams.max_F_rm

    sample_rate = hparams.sample_rate
    T = 1 / sample_rate

    main()
