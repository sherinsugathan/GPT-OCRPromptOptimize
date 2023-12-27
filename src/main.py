import os
import csv
import pandas as pd
import shutil
import glob
import openai
import Levenshtein
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import CountVectorizer
import difflib
import re
import time

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


data_root = "C:/Sherin/Workspace/2_Datasets/OCR_dataset/"
gcv_file = data_root + "all_data.csv"
aocr_file = data_root + "cjeu_files/FR/FrenchAnalysis/fileSelection3/pageSelection/100Files/OCR/RTF/TXT/"
gpt_result_folder = aocr_file + "/GPT/"

usecols_gcv = ["image_file", "original_text", "current_text", "language", "coded"]
df = pd.read_csv(gcv_file, usecols=usecols_gcv)

old_distance_list = []
new_distance_list = []
# List all files and directories in the specified path
file_list = os.listdir(aocr_file)
i = 0

#print(file_list)
#print(file_list[7:])
#quit()
#file_list = file_list[95:]

# Print the list of filenames
for fileName in file_list:
    i = i + 1
    while True:
        try:
            #fileName = "ECLI_EU_C_1955_9_1955_07_19_EN_4.txt"
            image_name = fileName[:-3] + "jpg"
            result_file = gpt_result_folder + fileName

            extractedRow = df[df['image_file']==image_name]
            #print(extractedRow)
            column_headers = extractedRow.columns.values.tolist()
            #print("The Column Header :", column_headers)

            gcv_raw_text = extractedRow.iloc[0, 2].lower()
            gcv_corrected_text = extractedRow.iloc[0, 3].lower()

            # Read the content of the text file
            file_path = aocr_file + fileName
            with open(file_path, 'r', encoding='utf-8') as file:
                text_to_correct = file.read()

            openai.api_key = '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'

            corrected_text = fix_cjeu(text_to_correct)

            # Example usage:
            distance_old = normalized_levenshtein_distance(preprocess_text(gcv_raw_text), preprocess_text(gcv_corrected_text))
            distance_new = normalized_levenshtein_distance(preprocess_text(corrected_text), preprocess_text(gcv_corrected_text))

            old_distance_list.append(distance_old)
            new_distance_list.append(distance_new)
            print(str(i), fileName, distance_old, distance_new)
            with open(result_file, 'w', encoding='utf-8') as file:
                file.write(corrected_text)
            break
        except Exception as e:
            print("Error occured. Retrying...")
            time.sleep(2)

print("###### COMPLETED ########")