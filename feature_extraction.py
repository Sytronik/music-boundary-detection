"""
feature_extraction.py

A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
"""
import os
from argparse import ArgumentParser
import numpy as np
import librosa
from pathlib import Path
import csv

from hparams import hparams


def melspectrogram(y, sr):
    S = librosa.stft(y,
                     n_fft=hparams.fft_size,
                     hop_length=hparams.hop_size,
                     win_length=hparams.win_size)

    if not hasattr(melspectrogram, 'mel_basis'):
        melspectrogram.mel_basis = librosa.filters.mel(sr,
                                                       n_fft=hparams.fft_size,
                                                       n_mels=hparams.num_mels)
    mel_S = np.dot(melspectrogram.mel_basis, np.abs(S))
    logmel_S = np.log10(1 + 10 * mel_S)
    logmel_S = logmel_S.T

    return logmel_S


def main(kind_data: str):
    # TODO: feature extraction
    path_metadata = hparams.path_dataset[kind_data] / 'metadata/metadata.csv'
    path_audio_dir = hparams.path_dataset[kind_data] / 'audio'
    path_annot_dir = hparams.path_dataset[kind_data] / 'annotations'
    songs = []
    sr = 44100
    T = 1 / sr

    f_metadata = path_metadata.open('r', newline='')
    for idx, l_meta in enumerate(csv.reader(f_metadata)):
        if idx == 0:
            continue
        song_id = int(l_meta[0])

        path_audio = path_audio_dir / f'{song_id}.mp3'
        if path_audio.exists():
            songs += song_id
        audio, _ = librosa.load(str(path_audio), sr=sr, mono=False)
        length = audio.shape[1]
        mel = melspectrogram(audio, sr)
        np.save(hparams.path_feature[kind_data] / f'{song_id}.npy', mel)

        n_frames = np.arange(0, length, hparams.hop_size)
        t_frames = n_frames * T

        t_boundaries = []
        sections = []
        f_annot = (
            path_annot_dir / f'{song_id}/parsed/textfile1_uppercase.txt'
        ).open('r', newline='')
        for l_annot in csv.reader(f_annot, delimiter='\t'):
            t_boundaries += l_annot[0]
            sections += l_annot[1]
        f_annot.close()

        # TODO: boundary label / segmentation map per song basis / segmap by section names
        # boundary_labels = []
        # raw_segmap = []
        # i_boundary = 0
        # for i_frame in range(len(t_frames)):
        #     t_frames[i_frames]

    f_metadata.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('kind_data')
    args = parser.parse_args()
    main(args.kind_data)
