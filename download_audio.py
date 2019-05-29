from __future__ import unicode_literals
import youtube_dl
import librosa
import os
import numpy as np
from pathlib import Path
import decimal
import subprocess



def download_yt(row: list):
    file_name = './SALAMI/audio_youtube/' + csv_data[row][header.index('salami_id')] + '.mp3'
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': file_name,
        'quiet': True,
        'ignoreerrors': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    url = 'https://www.youtube.com/watch?v=' + csv_data[row][header.index('youtube_id')]

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        print(file_name + ' is successfully downloaded')


def match_salami(row: list):
    file_name = './SALAMI/audio_youtube/' + csv_data[row][header.index('salami_id')] + '.mp3'
    y, sr = librosa.load(file_name, sr=None)
    start_pad_time = decimal.Decimal(csv_data[row][header.index('onset_in_salami')]) \
                     - decimal.Decimal(csv_data[row][header.index('onset_in_youtube')])
    N_start_pad = int(start_pad_time * sr)
    N_length = round(float(csv_data[row][header.index('salami_length')]) * sr)

    if start_pad_time > 0:
        y = np.pad(y, (N_start_pad,), mode='constant')
    elif start_pad_time < 0:
        y = y[np.abs(N_start_pad):]

    save_path = './SALAMI/matched_audio'
    save_name = save_path + '/' + csv_data[row][header.index('salami_id')] + '.wav'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    librosa.output.write_wav(save_name, y, sr)
    converted_name = './SALAMI/matched_audio/' + csv_data[row][header.index('salami_id')] + '.mp3'

    subprocess.call(['ffmpeg', '-i',
                     save_name,
                     converted_name])
    os.remove(save_name)


csv_path = Path('.', 'salami_youtube_pairings.csv')

with csv_path.open('r') as file:
    csv_data = []
    for line in file.readlines():
        line = line.replace('\n', '')
        csv_data.append(line.split(','))

header = csv_data[0]
print(header)


for row in range(1, len(csv_data)):
    download_yt(row)


for row in range(1, len(csv_data)):
    try:
        match_salami(row)
    except:
        continue