"""
This script suppose there are song_name.mp3 and song_name.txt in 'path/original' folder.
This converts that file structure to the same structure as SALAMI.

"""
import os
import shutil
import csv
from pathlib import Path

path = Path('/soundlab-salami-test')
path_original = path / 'original'
path_metadata = path / 'metadata/metadata.csv'
path_audio = path / 'audio'
shutil.rmtree(path_audio)
path_annot = path / 'annotations'
shutil.rmtree(path_annot)

# os.makedirs(path_metadata.parent)
os.makedirs(path_audio, exist_ok=True)
os.makedirs(path_annot, exist_ok=True)

mp3_list = sorted(list(path_original.glob('*.mp3')))
txt_list = sorted(list(path_original.glob('*.txt')))

with path_metadata.open('w') as f_metadata:
    writer = csv.writer(f_metadata)
    writer.writerow(['SONG_ID', 'SONG_TITLE', 'ARTIST', 'CLASS', 'SONG_WAS_DISCARDED_FLAG'])
    for idx, (path_mp3, path_txt) in enumerate(zip(mp3_list, txt_list)):
        writer.writerow([idx, path_mp3.stem, '', '', 'FALSE'])

        shutil.copy(str(path_mp3), str(path_audio))
        os.rename(path_audio / path_mp3.name, path_audio / f'{idx}.mp3')

        path_annot_idx = path_annot / str(idx) / 'parsed'
        os.makedirs(path_annot_idx)
        shutil.copy(str(path_txt), str(path_annot_idx))
        os.rename(path_annot_idx / path_txt.name, path_annot_idx / 'textfile1_uppercase.txt')
