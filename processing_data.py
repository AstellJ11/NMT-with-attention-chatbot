import json
import tqdm as tqdm
import os
import sys
import logging
from pathlib import Path
from downloading_data import init_logging

init_logging()
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))  # Access the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

train_dir = "raw_data/dstc8-schema-guided-dialogue/train"
all_dialogs = []  # Create an empty list array

# Create processed_data folder for all dialogue
new_dir = Path(current_dir, "processed_data/train")

try:
    os.makedirs(new_dir)
except OSError:
    logger.info("Creation of the directory '%s' failed!" % new_dir)
else:
    logger.info("Successfully created the directory '%s'!" % new_dir)

for file_name in tqdm.tqdm(os.listdir(train_dir), desc='Extracting utterances'):  # Display progress bar during loop

    if 'schema.json' in file_name:
        continue

    file_path = os.path.join(train_dir, file_name)  # Create new file path for each run of loop

    # Open each dialogue file
    with open(file_path, "r") as f:
        data = json.load(f)

    part_dialogs = []  # Create an empty list for each dialogue to be added too

    # for dialogue in data:
    #     theme = [dialogue['services']]
    #     if theme not in part_dialogs:
    #         part_dialogs.extend(theme)
    # all_dialogs.extend(part_dialogs)  # Add all elements of new dialogue to overall list

    # CURRENTLY WORKS AND EXTRACTS ALL 'Restaurants_1' EVEN IF MULTIPLE DOMAINS
    # Look inside .json for utterance only in 'Restaurants_1'
    for dialogue in data:
        if 'Restaurants_1' in dialogue['services']:
            for item in dialogue['turns']:
                utterance = [item['utterance']]  # Extract the system and user speech
                part_dialogs.extend(utterance)
    all_dialogs.extend(part_dialogs)  # Add all elements of new dialogue to overall list

# res_list = []
#
# for item in all_dialogs:
#     if item not in res_list:
#         res_list.append(item)
#
# print("Unique elements of the list using append():\n")
# for item in res_list:
#     print(item)


delim = "\n"  # initializing delimiter
temp = list(map(str, all_dialogs))  # Convert each list element to a string

res = delim.join(temp)  # Add each individual utterance to a new line

lines = res.splitlines()  # Split on new line

# For every utterance create a tab break, for every other utterance create a new line
processed = ''
step_size = 2
for i in range(0, len(lines), step_size):
    processed += '\t'.join(lines[i:i + step_size]) + '\n'

# File Saving
file_path = Path(current_dir, "processed_data/train/all_training_dialogue.txt")
logger.info(f"Saving Schema dialogue data to {file_path}")

save_path = open("processed_data/train/all_training_dialogue.txt", "w")
n = save_path.write(str(processed))
save_path.close()
logger.info("Processing Complete!")
