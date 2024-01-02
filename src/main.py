#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Name: main.py
Description: Script to run a specific GPT prompt for a specific engine to generate the .
Author: Sherin Sugathan
Created Date: 26-12-2023
Last Modified: 29-12-2023

Usage: python main.py [arguments]

License: GPL
"""

import os
import csv
import pandas as pd
import shutil
import glob
import openai
import Levenshtein
import difflib
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import zipfile
import numpy as np
from scipy.interpolate import make_interp_spline

def preprocess_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove new lines
    text = re.sub(r'\n', ' ', text)
    # Remove spaces at the beginning and end of the text
    text = text.lower().strip()
    return text

def extract_zip(zip_path):
    """
    Extracts a zip file to the same directory where the zip file is located.
    :param zip_path: Path to the zip file.
    """
    # Determine the directory where the zip file is located
    extract_to = os.path.dirname(zip_path)
    print("Please wait! Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def normalized_levenshtein_distance(s1, s2):
    if not s1 and not s2:  # Both strings are empty
        return 0
    distance = Levenshtein.distance(s1, s2)
    length = max(len(s1), len(s2))
    normalized_distance = distance / length
    return normalized_distance


def proc_text4_notworking(text_in):
    # Call GPT-4 chat
    theprompt = prompt_prefix + '\n""""' + text_in + '"""'
    tokenCount = len(prompt_prefix.split()) + len(text_in.split())
    openai.api_version = "2023-07-01-preview"
    response = openai.ChatCompletion.create(
        engine="cjeu-4",
        prompt=theprompt,
        temperature=0.0,
        max_tokens=tokenCount + 100,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    gpt_out = response.choices[0].message.text
    #return gpt_out, response, theprompt
    return gpt_out


def proc_text4(text_in):
    theprompt = prompt_prefix + '\n""""' + text_in + '"""'
    #system_prompt = "Consider the OCR output text and make corrections. The language is French."
    openai.api_version = "2023-07-01-preview"
    response = openai.ChatCompletion.create(
        engine="cjeu-4",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": theprompt}],
        temperature=0.0,
        max_tokens=4000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    gpt_out = response.choices[0].message.content
    return gpt_out

def proc_text_instruct(text_in):
    # Call GPT-3.5 instruct
    tokenCount = len(prompt_prefix.split()) + len(text_in.split())
    #system_prompt = "Consider the OCR output text and make corrections. The language is French."
    theprompt = system_prompt + "\n" + prompt_prefix + '\n""""' + text_in + '"""'
    tokenCount = len(prompt_prefix.split()) + len(text_in.split())
    openai.api_version = "2023-05-15"
    response = openai.Completion.create(
        engine="cjeu-35-turbo-instruct",
        prompt=theprompt,
        temperature=0,
        max_tokens=9999,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        stop=None)

    gpt_out = response.choices[0].text
    #return gpt_out, response, theprompt
    return gpt_out


parser = argparse.ArgumentParser(description='Run a prompt on cjeu french text.')
# Adding arguments
parser.add_argument('engine', type=str, choices=['cjeu-35-turbo-instruct', 'cjeu-4'], help='Name of the GPT Engine')
parser.add_argument('sysprompt', type=str, help='The system prompt string')
parser.add_argument('userprompt', type=str, help='The user prompt string')
parser.add_argument('--sample', type=int, default=100, help='Input Sampling percentage value')
parser.add_argument('--indices', nargs='*', type=int,  help='A list of specific file indices.')

# Parsing arguments
args = parser.parse_args()
#prompt_prefix = "Please note that the text to be corrected is in French. Fix spelling mistakes, do not add/remove words, make consistent word spacing, add missing spaces, fix font case issues within words, fix numbering issues, make consistent line breaks for the following text:"
system_prompt = args.sysprompt
prompt_prefix = args.userprompt
engineSelection = args.engine
percentage_to_process = args.sample

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
data_root = str(parent_dir) + "/data/"
gcv_file = data_root + "all_data.csv"
aocr_file = data_root + "fr/"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
gpt_result_folder = aocr_file + "/GPT_" + timestamp + "/"
gpt_image_result_fileName = data_root + "GPT_" + timestamp + ".png"

zip_path = data_root + "all_data.zip"
zip_file_path = data_root + "all_data.csv"

if os.path.exists(zip_file_path) == False:
    # ground truth data extraction.
    if os.path.exists(zip_path):
        print(f"Zip file found: {zip_path}")
        # Extract the zip file
        extract_zip(zip_path)

# Set up API
openai.api_type = "azure"
openai.api_base = "https://gpt-cjeuscans-test-swe.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = os.environ.get("OPENAI_API_KEY")

usecols_gcv = ["image_file", "original_text", "current_text", "language", "coded"]
df = pd.read_csv(gcv_file, usecols=usecols_gcv)

old_distance_list = []
new_distance_list = []
# List all files and directories in the specified path
file_list = [f for f in os.listdir(aocr_file) if os.path.isfile(os.path.join(aocr_file, f))]
i = 0

if args.indices:
        file_list = [file_list[i] for i in args.indices]

number_of_files_to_process = int(len(file_list) * (percentage_to_process / 100))
if(number_of_files_to_process == 0):
    print("Error: Insufficient number of input samples. Please check you arguments.")
    quit()

# Check if the folder exists
if not os.path.exists(gpt_result_folder):
    # Create the folder
    os.makedirs(gpt_result_folder)
else:
    print(f"Folder already exists: {gpt_result_folder}")

# Print the list of filenames
for fileName in file_list[:number_of_files_to_process]:
    i = i + 1
    while True:
        try:
            # Read the content of the text file
            file_path = aocr_file + fileName
            if (i > len(file_list[:number_of_files_to_process])):
                break
            image_name = fileName[:-3] + "jpg"
            result_file = gpt_result_folder + fileName
            distance_result_file = gpt_result_folder + "result.txt"
            error_stat_file = gpt_result_folder + "error_stat.txt"
            extractedRow = df[df['image_file']==image_name]
            column_headers = extractedRow.columns.values.tolist()

            gcv_raw_text = extractedRow.iloc[0, 2].lower()
            gcv_corrected_text = extractedRow.iloc[0, 3].lower()

            with open(file_path, 'r', encoding='utf-8') as file:
                text_to_correct = file.read()

            if(engineSelection == "cjeu-35-turbo-instruct"):
                # perform gpt based correction.
                corrected_text = proc_text_instruct(text_to_correct)
            if(engineSelection == "cjeu-4"):
                # perform gpt based correction.
                corrected_text = proc_text4(text_to_correct)

            # Example usage:
            distance_old = normalized_levenshtein_distance(preprocess_text(gcv_raw_text), preprocess_text(gcv_corrected_text))
            distance_new = normalized_levenshtein_distance(preprocess_text(corrected_text), preprocess_text(gcv_corrected_text))

            old_distance_list.append(distance_old)
            new_distance_list.append(distance_new)
            print(str(i), fileName, distance_old, distance_new)
            output_line = f"{i} {fileName} {distance_old} {distance_new}\n"
            with open(distance_result_file, "a") as file:
                file.write(output_line)
            with open(result_file, 'w', encoding='utf-8') as file:
                file.write(corrected_text)
            break
        except Exception as e:
            print("Error occurred. Retrying...")
            time.sleep(2)

# Plotting
df = pd.read_csv(distance_result_file, delim_whitespace=True, header=None)
# Assigning columns to variables
X = df.iloc[:, 0]
Y1 = df.iloc[:, 2]
Y2 = df.iloc[:, 3]

# Plotting Y1 and Y2 over X
plt.figure(figsize=(30, 6))
plt.plot(X, Y1, label='Google Vision', color='blue', linestyle='-', marker='o')
plt.plot(X, Y2, label='Adobe+GPT', color='green', linestyle='-', marker='o')
plt.xlabel('X')
plt.ylabel('Editing Distance')
plt.title('Normalized Levenshtein Distance comparison between Google Vision and Adobe+GPT (' + str(engineSelection) + ')')
plt.legend()
even_X_labels = [x for x in X if x % 2 == 0]
plt.xticks(even_X_labels)
plt.ylim(0, 1)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#plt.xticks(X)
#plt.grid(True)
plt.savefig(gpt_image_result_fileName, format='png', dpi=300, bbox_inches='tight', transparent=False, orientation ='landscape')
plt.show()

# Calculate basic statistics
sum_error = np.sum(Y2)
mean_error = np.mean(Y2)
median_error = np.median(Y2)
std_dev_error = np.std(Y2)
variance_error = np.var(Y2)

with open(error_stat_file, 'w') as file:
    file.write(f"Sum of Errors: {sum_error}\n")
    file.write(f"Mean Error: {mean_error}\n")
    file.write(f"Median Error: {median_error}\n")
    file.write(f"Standard Deviation: {std_dev_error}\n")
    file.write(f"Variance: {variance_error}\n")

print(f"Textual Result Location: {gpt_result_folder}")
print(f"Graphical Result Location: {gpt_image_result_fileName}")
print("###### COMPLETED ########")