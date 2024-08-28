import os
import json
import regex as re
from tqdm import tqdm

# Define the preprocessing functions
def preprocess_text(text):
    text = str(text)

    # 1. Replace '\n' or '  ' with ' '
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')

    # 2. Replace 'ㅇㅇ' with '응'
    text = re.sub(r'(?<=^|\s)ㅇ{2,}(?=$|\s)', '응', text)
    
    # 3. Replace special characters
    text = re.sub(r"\.{4,}", '...', text)
    text = re.sub(r"\?{2,}", '??', text)

    # 4. Remove special characters
    special_characters = '~!><@_/-ᆢ:;()^'
    for char in special_characters:
        text = text.replace(char, '')
    
    # 5. Remove text matching regex patterns
    text = re.sub(r'\b안녕\w*', '', text)
    text = re.sub(r'\b반갑습\w*', '', text)
    
    # 6. Replace "name1" with "화자1" and "name2" with "화자2"
    text = text.replace('name1', '화자1').replace('name2', '화자2')
    
    # 7. Remove consonants/vowels only text like "ㅋㅋㅋ", "ㅎㅎㅎ", "ㅠㅠㅠ"
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', '', text)

    # 8. If the text contains no Korean and only special characters, replace with empty string
    if not re.search(r'[가-힣]', text) and re.search(r'[^\w\s]', text):
        text = ''
    
    return text.strip()

# Function to merge consecutive utterances of the same speaker
def merge_consecutive_utterances(conversation, reference_id):
    merged_conversation = []
    previous_speaker = None
    previous_utterance = ''
    previous_utterance_ids = []

    for utterance in conversation:
        speaker = utterance['speaker']
        text = utterance['utterance']
        utterance_id = utterance['utterance_id']

        if speaker == previous_speaker and utterance_id not in reference_id and previous_utterance_ids[-1] not in reference_id:
            # Merge with previous utterance
            previous_utterance += ' ' + text
            previous_utterance_ids.append(utterance_id)
        else:
            if previous_speaker is not None and previous_utterance.strip() != '':
                # Append the previous utterance to the merged conversation
                merged_conversation.append({
                    'speaker': previous_speaker,
                    'utterance': previous_utterance.strip(),
                    'utterance_id': previous_utterance_ids if len(previous_utterance_ids) > 1 else previous_utterance_ids[0]
                })
            # Start a new utterance
            previous_speaker = speaker
            previous_utterance = text
            previous_utterance_ids = [utterance_id]

    # Add the last accumulated utterance
    if previous_speaker is not None and previous_utterance.strip() != '':
        merged_conversation.append({
            'speaker': previous_speaker,
            'utterance': previous_utterance.strip(),
            'utterance_id': previous_utterance_ids if len(previous_utterance_ids) > 1 else previous_utterance_ids[0]
        })

    return merged_conversation

# Define input and output directories
input_data_dir = 'resource/data'
output_data_dir = 'resource/refined_data'

if __name__=="__main__":
    # Ensure the output directory exists
    os.makedirs(output_data_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_data_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(input_data_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            
            output_json_data = []
            for datum in json_data:
                # Preprocess each conversation's utterance and remove empty utterances
                preprocessed_conversation = []
                for conversation in datum['input']['conversation']:
                    conversation['utterance'] = preprocess_text(conversation['utterance'])
                    if conversation['utterance'] != '':
                        preprocessed_conversation.append(conversation)
                
                # Merge consecutive utterances
                datum['input']['conversation'] = merge_consecutive_utterances(preprocessed_conversation, datum['input']['reference_id'])
                
                output_json_data.append(datum)

            # Define the output file path
            output_filepath = os.path.join(output_data_dir, filename)
            
            # Write the processed data to the output file
            with open(output_filepath, 'w', encoding='utf-8') as file:
                json.dump(output_json_data, file, ensure_ascii=False, indent=4)

    print("Preprocessing, merging, and filtering complete.")