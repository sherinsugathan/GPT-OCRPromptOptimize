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


def proc_text4(text_in):
    # Call GPT-4 chat
    theprompt = prompt_prefix + '\n""""' + text_in + '"""'
    openai.api_version = "2023-07-01-preview"
    response = openai.ChatCompletion.create(
        engine="cjeu-4",
        messages=[{"role": "user", "content": theprompt}],
        temperature=0.0,
        max_tokens=4000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    gpt_out = response.choices[0].message.content
    return gpt_out, response, theprompt

def proc_text_instruct(text_in):
    # Call GPT-3.5 instruct
    return "hello"
    theprompt = prompt_prefix + '\n""""' + text_in + '"""'
    response = openai.Completion.create(
        engine="cjeu-35-turbo-instruct",
        prompt=theprompt,
        temperature=0.1,
        max_tokens=4000,
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
parser.add_argument('prompt', type=str, help='The prompt string')
# Parsing arguments
args = parser.parse_args()
#prompt_prefix = "Please note that the text to be corrected is in French. Fix spelling mistakes, do not add/remove words, make consistent word spacing, add missing spaces, fix font case issues within words, fix numbering issues, make consistent line breaks for the following text:"
prompt_prefix = args.prompt
engineSelection = args.engine

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

# Check if the folder exists
if not os.path.exists(gpt_result_folder):
    # Create the folder
    os.makedirs(gpt_result_folder)
    print(f"Folder created: {gpt_result_folder}")
else:
    print(f"Folder already exists: {gpt_result_folder}")


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

# Print the list of filenames
for fileName in file_list:
    i = i + 1
    while True:
        try:
            # Read the content of the text file
            file_path = aocr_file + fileName

            if (i > len(file_list)):
                break
            image_name = fileName[:-3] + "jpg"
            result_file = gpt_result_folder + fileName
            distance_result_file = gpt_result_folder + "result.txt"
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
X = df.iloc[:, 0]   # First column
Y1 = df.iloc[:, 2]  # Third column
Y2 = df.iloc[:, 3]  # Fourth column

# Plotting Y1 and Y2 over X
plt.figure(figsize=(30, 6))
plt.plot(X, Y1, label='Google Vision', color='blue', linestyle='-', marker='o')
plt.plot(X, Y2, label='Adobe+GPT', color='green', linestyle='-', marker='o')
plt.xlabel('X')
plt.ylabel('Editing Distance')
plt.title('Normalized Levenshtein Distance comparison between Google Vision and Adobe+GPT')
plt.legend()
even_X_labels = [x for x in X if x % 2 == 0]
plt.xticks(even_X_labels)
#plt.xticks(X)
#plt.grid(True)
plt.savefig(gpt_image_result_fileName, format='png', dpi=300, bbox_inches='tight', transparent=False, orientation ='landscape')
plt.show()
print("###### COMPLETED ########")