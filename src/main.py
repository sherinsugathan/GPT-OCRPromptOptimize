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
from pathlib import Path
from datetime import datetime

def preprocess_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove new lines
    text = re.sub(r'\n', ' ', text)
    # Remove spaces at the beginning and end of the text
    text = text.lower().strip()
    return text

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

def fix_cjeu(text):
    # Construct a chat-based prompt for ChatGPT
    message = {
        "role": "system",
        "content": "Consider the OCR output text and make corrections. The language is French."
    }
    user_message = {
        "role": "user",
        "content": f"Please note that the text to be corrected is in French. Fix spelling mistakes, do not add/remove words, make consistent word spacing, add missing spaces, fix font case issues within words, fix numbering issues, make consistent line breaks for the following text: '{text}'"
    }

    # Use the Chat Completion endpoint of the API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # ChatGPT model
        messages=[message, user_message]
    )

    # Extract and return the assistant's response
    assistant_message = response.choices[0].message['content']
    return assistant_message

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
data_root = str(parent_dir) + "/data/"
gcv_file = data_root + "all_data.csv"
aocr_file = data_root + "fr/"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
gpt_result_folder = aocr_file + "/GPT_" + timestamp + "/"

# Check if the folder exists
if not os.path.exists(gpt_result_folder):
    # Create the folder
    os.makedirs(gpt_result_folder)
    print(f"Folder created: {gpt_result_folder}")
else:
    print(f"Folder already exists: {gpt_result_folder}")

prompt_prefix = "Please note that the text to be corrected is in French. Fix spelling mistakes, do not add/remove words, make consistent word spacing, add missing spaces, fix font case issues within words, fix numbering issues, make consistent line breaks for the following text:"
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

            # perform gpt based correction.
            corrected_text = proc_text_instruct(text_to_correct)

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

print("###### COMPLETED ########")