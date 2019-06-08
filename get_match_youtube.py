from argparse import ArgumentParser
from decimal import Decimal
import os
import csv
import subprocess
from pathlib import Path
from typing import List, Dict

import librosa
import numpy as np
import youtube_dl


def download_yt(row: Dict[str, str]):
    s_fname = str(path_raw_audio / f'{row["salami_id"]}.mp3')
    ydl_opts = dict(format='bestaudio/best',
                    outtmpl=s_fname,
                    quiet=True,
                    ignoreerrors=True,
                    postprocessors=[dict(key='FFmpegExtractAudio',
                                         preferredcodec='mp3',
                                         preferredquality='192')],
                    )

    url = f'https://www.youtube.com/watch?v={row["youtube_id"]}'

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        print(f'{s_fname} is successfully downloaded')


def match_salami(row: Dict[str, str]):
    spath_original = str(path_raw_audio / f'{row["salami_id"]}.mp3')
    y, sr = librosa.load(spath_original, sr=None)
    start_pad_time = (Decimal(row['onset_in_salami'])
                      - Decimal(row['onset_in_youtube']))
    N_start_pad = round(start_pad_time * sr)
    # N_length = round(float(row['salami_length']) * sr)

    if start_pad_time > 0:
        y = np.pad(y, (N_start_pad,), mode='constant')
    elif start_pad_time < 0:
        y = y[np.abs(N_start_pad):]

    path_wav = path_matched_audio / f'{row["salami_id"]}.wav'

    librosa.output.write_wav(str(path_wav), y, sr)
    path_mp3 = path_wav.with_suffix('.mp3')

    subprocess.call(('ffmpeg', '-i', str(path_wav), str(path_mp3)))
    os.remove(path_wav)


def main(mode: str):
    path_csv = Path('./salami_youtube_pairings.csv')

    with path_csv.open('r') as f:
        csv_data = [line for line in csv.reader(f)]

    header = csv_data[0]
    print(header)
    rows_as_dict: List[Dict[str, str]] = []

    for row in csv_data[1:]:
        rows_as_dict.append({c: row[i] for i, c in enumerate(header)})

    if mode == 'download':
        for row in rows_as_dict:
            download_yt(row)
    else:
        for row in rows_as_dict:
            match_salami(row)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mode', choices=('download', 'match'))
    parser.add_argument('--path-dl', default='/salami-data-public/audio_youtube')
    parser.add_argument('--path-match', default='/salami-data-public/audio_matched')
    args = parser.parse_args()
    path_raw_audio = Path(args.path_dl)
    path_matched_audio = Path(args.path_match)
    main(args.mode)
