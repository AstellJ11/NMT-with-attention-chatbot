import json
import tqdm as tqdm
import os, sys
import logging

from downloading_data import init_logging

init_logging()
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))  # Access the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def add_dialogue_index(prefix, all_dialogue):  # TO DO
    return {
        prefix + "_" + str(idx): all_dialogue[idx]
        for idx in range(len(all_dialogue))
    }


train_dir = "data/dstc8-schema-guided-dialogue/train"
all_dialogs = []  # Create an empty list array outside of loop

for file_name in tqdm.tqdm(os.listdir(train_dir), desc='Processing'):  # Display progress bar during loop

    if 'schema.json' in file_name:
        continue

    file_path = os.path.join(train_dir, file_name)  # Create new file path for each run of loop

    with open(file_path, "r") as f:
        data = json.load(f)  # Open each dialog file

    part_dialogs = []  # Create an empty list for each dialog to be added too

    for dialog in data:
        one_dialog = [[item['speaker'].capitalize(), item['utterance']] for item in dialog['turns']]  # TO DO
        part_dialogs.append(one_dialog)  # Add the separated dialog to the end of list

    all_dialogs.extend(part_dialogs)  # Add all elements of new dialog to overall list

all_dialogs = add_dialogue_index("schema_dialog", all_dialogs)  # Call the add_dialogue_index function

file_path = os.path.join(current_dir, "data/all_training_dialog.json")  # Defines the new list .json file
logger.info(f"Saving Schema Dialog data to {file_path}")
with open(file_path, "w") as fp:
    json.dump(all_dialogs, fp, indent=4)  # Saving the .json file
logger.info("Processing Complete!")
