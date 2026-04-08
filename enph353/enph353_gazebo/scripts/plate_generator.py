#!/usr/bin/env python3
# https://phas.ubc.ca/~miti/ENPH353/ENPH353Clues.csv

import cv2
import csv
import numpy as np
import os
import random
import requests
import string
from random import randint
from PIL import Image, ImageFont, ImageDraw

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

# Set to True to load from a local CSV file instead of the URL.
# The local file must be in the same folder as this script and named
# LOCAL_CSV_FILENAME. It must have the same header row as the URL CSV,
# but can have multiple data rows — one is chosen at random.
USE_LOCAL_CSV    = True
LOCAL_CSV_FILENAME = "clues.csv"

# ------------------------------------------------------------------ #

def loadCrimesProfileCompetition():
    '''
    @brief returns a set of clues for one game and saves them to plates.csv
    @retval clue dictionary of the form
                [size:value, victim:value, ...
                 crime:value, time:value,
                 place:value, motive:value,
                 weapon:value, bandit:value]
    '''
    if USE_LOCAL_CSV:
        local_path = os.path.join(SCRIPT_PATH, LOCAL_CSV_FILENAME)
        print(f"Loading clues from local file: {local_path}")
        with open(local_path, 'r') as f:
            reader = csv.reader(f)
            rows   = list(reader)

        # First row is the header, remaining rows are data
        key_list   = rows[0]
        data_rows  = rows[1:]

        if not data_rows:
            raise ValueError(f"No data rows found in {LOCAL_CSV_FILENAME}")

        # Pick a random row
        value_list = random.choice(data_rows)
        print(f"Selected row: {value_list}")
    else:
        URL = "https://phas.ubc.ca/~miti/ENPH353/ENPH353Clues.csv"
        print("Loading clues from URL...")
        response  = requests.get(URL)
        raw       = response.text.split('\n')
        key_list  = raw[0].split(',')
        value_list = raw[1].split(',')

    clues = {}
    with open(SCRIPT_PATH + "plates.csv", 'w') as plates_file:
        csvwriter = csv.writer(plates_file)
        for (key, value) in zip(key_list, value_list):
            clues[key] = value.strip().upper()
            csvwriter.writerow([key, value.strip().upper()])

    return clues


# Find the path to this script
SCRIPT_PATH  = os.path.dirname(os.path.realpath(__file__)) + "/"
TEXTURE_PATH = '../media/materials/textures/'

banner_canvas = cv2.imread(SCRIPT_PATH + 'clue_banner.png')
PLATE_HEIGHT  = 600
PLATE_WIDTH   = banner_canvas.shape[1]
IMG_DEPTH     = 3

clues = loadCrimesProfileCompetition()

i = 0
for key, value in clues.items():
    entry = key + "," + value
    print(entry)

    blank_plate_pil = Image.fromarray(banner_canvas)
    draw = ImageDraw.Draw(blank_plate_pil)

    font_size  = 90
    monospace  = ImageFont.truetype(
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", font_size)
    font_color = (255, 0, 0)

    draw.text((250, 30),  key,   font_color, font=monospace)
    draw.text((30,  250), value, font_color, font=monospace)

    populated_banner = np.array(blank_plate_pil)
    cv2.imwrite(
        os.path.join(SCRIPT_PATH + TEXTURE_PATH + "unlabelled/",
                     "plate_" + str(i) + ".png"),
        populated_banner)
    i += 1