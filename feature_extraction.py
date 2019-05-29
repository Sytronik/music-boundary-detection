"""
feature_extraction.py

A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
"""
import csv
from argparse import ArgumentParser
from itertools import product

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
        S += librosa.stft(item_y,
                          n_fft=hparams.fft_size,
                          hop_length=hparams.hop_size,
                          win_length=hparams.win_size)  # (F[linear], T)

    S = np.stack(S, axis=0)  # (k, F[linear], T)

    if not hasattr(melspectrogram, 'mel_basis'):
        melspectrogram.mel_basis = librosa.filters.mel(sr,
                                                       n_fft=hparams.fft_size,
                                                       n_mels=hparams.num_mels)
    mel_S = np.einsum('mf,kft->kmt', melspectrogram.mel_basis, np.abs(S))  # k, F[mel], T
    logmel_S = np.log10(1 + 10 * mel_S)
    logmel_S = logmel_S.squeeze()  # (1, F, T) -> (F, T)

    return logmel_S


def main(kind_data: str):
    path_metadata = hparams.path_dataset[kind_data] / 'metadata/metadata.csv'
    path_audio_dir = hparams.path_dataset[kind_data] / 'audio'
    path_annot_dir = hparams.path_dataset[kind_data] / 'annotations'

    pitchstep = (0, *hparams.pitchstep)
    noise_db = (None, *hparams.noise_db)
    max_F_rm = (0, *hparams.max_F_rm)

    songs = []
    sect_names = {'Silence'}
    sect_maps = dict()
    coarse_maps = dict()
    sr = 44100
    T = 1 / sr

    f_metadata = path_metadata.open('r', newline='')
    pbar = tqdm(csv.reader(f_metadata), dynamic_ncols=True)
    for idx, l_meta in enumerate(pbar):
        if idx == 0:
            continue
        song_id = int(l_meta[0])

        path_audio = path_audio_dir / f'{song_id}.mp3'
        if path_audio.exists():
            songs += song_id
        y, _ = librosa.load(str(path_audio), sr=sr, mono=False)  # 2, n
        length = y.shape[1]

        if kind_data == 'train':
            ys_pitch = {0: y}
            for step, db, F in product(pitchstep, noise_db, max_F_rm):
                # pitch shift
                if step not in ys_pitch:
                    ys_pitch[step] = librosa.effects.pitch_shift(y, sr, n_steps=step)
                y = ys_pitch[step]

                # adding noise
                if db is not None:
                    y += librosa.db_to_amplitude(db) * np.random.randn(*y.shape)

                mel = melspectrogram(y, sr)  # 2, F, T

                # SpecAugmentation
                if F != 0:
                    height = np.random.randint(1, F + 1)
                    f0 = np.random.randint(0, hparams.num_mels - height)
                    mel[..., f0:f0 + height, :] = 0

                np.save(hparams.path_feature[kind_data] / f'{song_id}.npy', mel)
        else:
            mel = melspectrogram(y, sr)
            np.save(hparams.path_feature[kind_data] / f'{song_id}.npy', mel)

        n_frames = np.arange(0, length, hparams.hop_size)
        t_frames = n_frames * T

        t_boundaries = []
        sections = []
        coarse = []
        f_section = (
                path_annot_dir / f'{song_id}/parsed/textfile1_functions.txt'
        ).open('r', newline='')
        f_coarse = (
                path_annot_dir / f'{song_id}/parsed/textfile1_uppercase.txt'
        ).open('r', newline='')
        iter_sect = csv.reader(f_section, delimiter='\t')
        iter_coarse = csv.reader(f_coarse, delimiter='\t')
        if len(iter_sect) != len(iter_coarse):
            raise Exception(song_id)
        for l_sect, l_coarse in zip(iter_sect, iter_coarse):
            t = float(l_sect[0])
            s = l_sect[1]
            c = l_coarse[1]
            if t == 0:
                if s != 'Silence' or c != 'Silence':
                    raise Exception(song_id)
                continue
            if s == 'End':
                continue
            if not sections:
                if sections[-1] == s and coarse[-1] == c:
                    continue
                elif sections[-1] == s ^ coarse[-1] == c:
                    raise Exception(song_id)
            t_boundaries += t
            sections += s
            coarse += c
        f_section.close()
        f_coarse.close()

        sect_names.union(sections)

        # boundary label / section segmentation map / coarse structure map
        boundary_labels = []
        sect_map = ['Silence']
        coarse_map = ['Silence']
        i_boundary = 0
        for i_frame in range(len(t_frames)):
            if i_boundary == len(t_boundaries):
                boundary_labels += 0
                sect_map += sect_map[-1]
                coarse_map += coarse_map[-1]
            elif t_boundaries[i_boundary] - t_frames[i_frame] > hparams.hop_size * T:
                boundary_labels += 0
                sect_map += sect_map[-1]
                coarse_map += coarse_map[-1]
            elif (t_boundaries[i_boundary] - t_frames[i_frame]
                  > t_frames[i_frame] - t_boundaries[i_boundary+1]):
                # if the next frame is closer than the current frame
                boundary_labels += 0
                sect_map += sect_map[-1]
                coarse_map += coarse_map[-1]
            elif (t_boundaries[i_boundary] - t_frames[i_frame]
                  <= t_frames[i_frame] - t_boundaries[i_boundary+1]):
                # if the current frame is closer than the next frame
                boundary_labels += 1
                sect_map += sections[i_boundary]
                coarse_map += coarse[i_boundary]
                i_boundary += 1
            elif (t_boundaries[i_boundary] - t_frames[i_frame]
                  < t_frames[i_frame] - t_boundaries[i_boundary-1]):
                # if the current frame is closer than the prev frame
                boundary_labels += 1
                sect_map += sections[i_boundary]
                coarse_map += coarse[i_boundary]
                i_boundary += 1
            else:
                raise Exception(song_id)

        sect_maps[song_id] = sect_map
        coarse_maps[song_id] = coarse_map

    f_metadata.close()

    dict_sect_idx = {name: idx for idx, name in enumerate(sect_names)}
    for song_id in sect_maps.keys():
        sect_maps[song_id] = np.array([dict_sect_idx[s] for s in sect_maps[song_id]])
        coarse_map = []
        dict_coarse_idx = {}
        idx = 0
        for c in coarse_maps[song_id]:
            if c not in dict_coarse_idx:
                dict_coarse_idx[c] = idx
                idx += 1
            coarse_map += dict_coarse_idx[c]
        coarse_maps[song_id] = np.array(coarse_map)

    np.savez(hparams.path_feature[kind_data] / 'section_maps.npz', **sect_maps)
    np.savez(hparams.path_feature[kind_data] / 'coarse_maps.npz', **coarse_maps)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('kind_data')
    args = parser.parse_args()
    main(args.kind_data)
