import json
import os
import sys
import logging
from downloading_data import init_logging

from pathlib import Path
import tqdm as tqdm
from pick import pick


def domain_choice():
    all_domains = []  # List array for domain names (including duplicates)
    unique_list = []  # List array to find unique domain names
    domain_names = []  # Final list array for all unique domains

    # Gather list of domains from dataset
    for file_name in tqdm.tqdm(os.listdir(train_dir), desc='Gathering domains'):  # Display progress bar during loop

        if 'schema.json' in file_name:
            continue

        file_path = os.path.join(train_dir, file_name)  # Create new file path for each run of loop

        # Open each dialogue file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract all domains from within 'services' dict
        temp_domains = []
        for dialogue in data:
            theme = [dialogue['services']]
            if theme not in temp_domains:
                temp_domains.extend(theme)
        all_domains.extend(temp_domains)

        # Extract all unique domain names
        for item in all_domains:
            if item not in unique_list:
                unique_list.append(item)

    # Convert list of list to flat base list
    flat_list = [item for sublist in unique_list for item in sublist]

    # Duplicates can still exist within list, final check and pass to domain_names
    for item in flat_list:
        if item not in domain_names:
            domain_names.append(item)

    # Allow the user to choose the domain being trained
    domain_question = 'Please choose a domain from the list you wish to talk about:  '
    domain_option, index = pick(domain_names, domain_question)

    return domain_option


def extract_utterance(inp_dir, out_dir):
    all_dialogs = []  # List array for final extracted dialogs

    for file_name in tqdm.tqdm(os.listdir(inp_dir), desc='Extracting utterances'):  # Display progress bar during loop

        if 'schema.json' in file_name:
            continue

        file_path = os.path.join(inp_dir, file_name)

        with open(file_path, "r") as f:
            data = json.load(f)

        temp_dialogs = []
        for dialogue in data:
            if domain_option in dialogue['services']:
                for item in dialogue['turns']:
                    utterance = [item['utterance']]  # Extract the system and user speech
                    temp_dialogs.extend(utterance)
        all_dialogs.extend(temp_dialogs)  # Add all elements of new dialogue to overall list

    # Process data into required format: I \t R \n I \t R...
    deli = "\n"  # Initialising delimiter
    temp_string = list(map(str, all_dialogs))  # Convert each list element to a string

    res = deli.join(temp_string)  # Add each individual utterance to a new line

    lines = res.splitlines()  # Split on new line

    # For every utterance create a tab break, for every other utterance create a new line
    processed = ''
    step_size = 2
    for i in range(0, len(lines), step_size):
        processed += '\t'.join(lines[i:i + step_size]) + '\n'

    # File Saving
    file_path = Path(current_dir, out_dir)
    logger.info(f"Saving Schema dialogue data to {file_path}")

    save_path = open(out_dir, "w")
    n = save_path.write(str(processed))
    save_path.close()
    logger.info("Processing Complete!")


def testing_translated_utterance(inp_dir, out_dir):
    all_dialogs = []  # List array for final extracted dialogs

    for file_name in tqdm.tqdm(os.listdir(inp_dir),
                               desc='Extracting testing utterances'):  # Display progress bar during loop

        if 'schema.json' in file_name:
            continue

        file_path = os.path.join(inp_dir, file_name)

        with open(file_path, "r") as f:
            data = json.load(f)

        temp_dialogs = []
        for dialogue in data:
            if domain_option in dialogue['services']:
                for item in dialogue['turns']:
                    utterance = [item['utterance']]  # Extract the system and user speech
                    temp_dialogs.extend(utterance)
        all_dialogs.extend(temp_dialogs)  # Add all elements of new dialogue to overall list

    all_dialogs.pop(0)  # Removes the first value to allow iteration to start at the second
    test_inp = all_dialogs[::2]  # Extract individual testing sentences from pairs

    # File Saving
    file_path = Path(current_dir, out_dir)
    logger.info(f"Saving Schema dialogue data to {file_path}")

    with open(file_path, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(test_inp))

    logger.info("Processing Complete!")


def testing_input_utterance(inp_dir, out_dir):
    all_dialogs = []  # List array for final extracted dialogs

    for file_name in tqdm.tqdm(os.listdir(inp_dir),
                               desc='Extracting testing utterances'):  # Display progress bar during loop

        if 'schema.json' in file_name:
            continue

        file_path = os.path.join(inp_dir, file_name)

        with open(file_path, "r") as f:
            data = json.load(f)

        temp_dialogs = []
        for dialogue in data:
            if domain_option in dialogue['services']:
                for item in dialogue['turns']:
                    utterance = [item['utterance']]  # Extract the system and user speech
                    temp_dialogs.extend(utterance)
        all_dialogs.extend(temp_dialogs)  # Add all elements of new dialogue to overall list

    test_inp = all_dialogs[::2]  # Extract individual testing sentences from pairs

    # File Saving
    file_path = Path(current_dir, out_dir)
    logger.info(f"Saving Schema dialogue data to {file_path}")

    with open(file_path, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(test_inp))

    logger.info("Processing Complete!")


# Initialise logger
init_logging()
logger = logging.getLogger(__name__)

# Access the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Unprocessed train + test directories
train_dir = "raw_data/dstc8-schema-guided-dialogue/train"
test_dir = "raw_data/dstc8-schema-guided-dialogue/test"

# Processed train + test directories
output_train_dir = "processed_data/train/all_training_dialogue.txt"
output_test_dir = "processed_data/test/all_testing_dialogue.txt"
output_test_dir2 = "processed_data/BLEU/human_translated_dialogue.txt"
output_test_dir3 = "processed_data/test/input_testing_dialogue.txt"

if __name__ == "__main__":
    # Create processed_data folders for all final dialogues
    new_train_dir = Path(current_dir, "processed_data/train")
    new_test_dir = Path(current_dir, "processed_data/test")
    new_test_dir2 = Path(current_dir, "processed_data/BLEU")
    try:
        os.makedirs(new_train_dir)
    except OSError:
        logger.info("Creation of the directory '%s' failed. It may already exist." % new_train_dir)
    else:
        logger.info("Successfully created the directory '%s'!" % new_train_dir)
    try:
        os.makedirs(new_test_dir)
    except OSError:
        logger.info("Creation of the directory '%s' failed. It may already exist." % new_test_dir)
    else:
        logger.info("Successfully created the directory '%s'!" % new_test_dir)
    try:
        os.makedirs(new_test_dir2)
    except OSError:
        logger.info("Creation of the directory '%s' failed. It may already exist." % new_test_dir2)
    else:
        logger.info("Successfully created the directory '%s'!" % new_test_dir2)

    # Gather user chosen domain
    domain_option = domain_choice()

    # Extract training + testing utterances
    extract_utterance(train_dir, output_train_dir)
    extract_utterance(test_dir, output_test_dir)  # Needed to add to vocab library
    testing_translated_utterance(test_dir, output_test_dir2)  # Human translated response for BLEU score
    testing_input_utterance(test_dir, output_test_dir3)
