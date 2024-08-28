import argparse
import csv
import json
import os
from collections import Counter
from PIL import Image 

import matplotlib.pyplot as plt
import numpy as np
import regex as re
from matplotlib import rc
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS

parser = argparse.ArgumentParser(prog="EDA")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--refined", action="store_true", help="whether use refined dataset")
args = parser.parse_args()

# Set font family to NanumGothic
rc('font', family='NanumGothic')

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if args.refined:
    EDA_RESULT_PATH = "resource/EDA/refined_data"
    train_data = load_json('resource/refined_data/train.json')
    dev_data = load_json('resource/refined_data/dev.json')
    trainable_data = load_json('resource/refined_data/train+dev.json')
    test_data = load_json('resource/refined_data/test.json')
else:
    EDA_RESULT_PATH = "resource/EDA"
    train_data = load_json('resource/data/train.json')
    dev_data = load_json('resource/data/dev.json')
    trainable_data = load_json('resource/data/train+dev.json')
    test_data = load_json('resource/data/test.json')

print(f"Train data size: {len(train_data)}")
print(f"Dev data size: {len(dev_data)}")
print(f"Trainable (Train+Dev) data size: {len(trainable_data)}")
print(f"Test data size: {len(test_data)}")

def basic_statistics(data):
    categories = [entry['input']['category'] for entry in data]
    outputs = [entry['output'] for entry in data]

    category_count = Counter(categories)
    output_count = Counter(outputs)

    print(f"Category distribution: {category_count}")
    print(f"Output distribution: {output_count}")

print("Train Data Statistics:")
basic_statistics(train_data)

print("\nDev Data Statistics:")
basic_statistics(dev_data)

print("\nTrainable (Train+Dev) Data Statistics:")
basic_statistics(trainable_data)

def conversation_analysis(data):
    conversation_lengths = []
    total_utterance_lengths = []

    for entry in tqdm(data, desc="Analyzing conversations"):
        conversation_lengths.append(len(entry['input']['conversation']))
        total_utterance_lengths.append(sum(len(str(utterance['utterance'])) for utterance in entry['input']['conversation']))

    avg_conversation_length = sum(conversation_lengths) / len(conversation_lengths)
    avg_total_utterance_length = sum(total_utterance_lengths) / len(total_utterance_lengths)

    print(f"Average conversation length (number of utterances): {avg_conversation_length}")
    print(f"Average total conversation length (number of characters): {avg_total_utterance_length}")

    return total_utterance_lengths

print("Train Data Conversation Analysis:")
train_total_utterance_lengths = conversation_analysis(train_data)

print("\nDev Data Conversation Analysis:")
dev_total_utterance_lengths = conversation_analysis(dev_data)

print("\nTrainable (Train+Dev) Data Conversation Analysis:")
dev_total_utterance_lengths = conversation_analysis(trainable_data)

print("\nTest Data Conversation Analysis:")
test_total_utterance_lengths = conversation_analysis(test_data)

def plot_distribution(counter, title, filename, total_count=None, sorted_keys=None):
    labels, values = zip(*counter.items())
    if sorted_keys:
        values = [counter[key] for key in sorted_keys]
        labels = sorted_keys
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    for bar in bars:
        height = bar.get_height()
        percentage = f"{(height/total_count*100):.2f}%" if total_count else ""
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n{percentage}', ha='center', va='bottom')
    os.makedirs(f'{EDA_RESULT_PATH}', exist_ok=True)
    plt.savefig(f'{EDA_RESULT_PATH}/{filename}.png', dpi=300)
    plt.close()

print("Train Data Category Distribution:")
plot_distribution(Counter([entry['input']['category'] for entry in train_data]), "Category Distribution (Train Data)", "train_category_distribution", total_count=len(train_data))

print("Train Data Output Distribution:")
plot_distribution(Counter([entry['output'] for entry in train_data]), "Output Distribution (Train Data)", "train_output_distribution", total_count=len(train_data), sorted_keys=["inference_1", "inference_2", "inference_3"])

print("Dev Data Category Distribution:")
plot_distribution(Counter([entry['input']['category'] for entry in dev_data]), "Category Distribution (Dev Data)", "dev_category_distribution", total_count=len(dev_data))

print("Dev Data Output Distribution:")
plot_distribution(Counter([entry['output'] for entry in dev_data]), "Output Distribution (Dev Data)", "dev_output_distribution", total_count=len(dev_data), sorted_keys=["inference_1", "inference_2", "inference_3"])

print("Trainable (Train+Dev) Data Category Distribution:")
plot_distribution(Counter([entry['input']['category'] for entry in trainable_data]), "Category Distribution (Train+Dev Data)", "train+dev_category_distribution", total_count=len(trainable_data))

print("Trainable (Train+Dev) Data Output Distribution:")
plot_distribution(Counter([entry['output'] for entry in trainable_data]), "Output Distribution (Train+Dev Data)", "train+dev_output_distribution", total_count=len(trainable_data), sorted_keys=["inference_1", "inference_2", "inference_3"])

print("Test Data Category Distribution:")
plot_distribution(Counter([entry['input']['category'] for entry in test_data]), "Category Distribution (Test Data)", "test_category_distribution", total_count=len(test_data))

def plot_histogram(data, title, filename, x_label, y_label, x_tick_step=1):
    plt.figure(figsize=(10, 5))
    counts, bins, bars = plt.hist(data, bins=range(0, max(data) + x_tick_step, x_tick_step), align='left', edgecolor='black')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(bins)
    min_val = np.min(data)
    mean_val = np.mean(data)
    max_val = np.max(data)
    plt.axvline(min_val, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean_val, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(max_val, color='b', linestyle='dashed', linewidth=1)
    plt.text(min_val, max(counts) * 0.9, f'Min: {min_val}', color='r', ha='center')
    plt.text(mean_val, max(counts) * 0.9, f'Mean: {mean_val:.2f}', color='g', ha='center')
    plt.text(max_val, max(counts) * 0.9, f'Max: {max_val}', color='b', ha='center')
    os.makedirs(f'{EDA_RESULT_PATH}', exist_ok=True)
    plt.savefig(f'{EDA_RESULT_PATH}/{filename}.png', dpi=300)
    plt.close()

print("Train Data Conversation Length Distribution:")
plot_histogram([len(entry['input']['conversation']) for entry in train_data], "Conversation Length Distribution (Train Data)", "train_conversation_length_distribution", "Length", "Frequency")

print("Train Data Total Utterance Length Distribution:")
plot_histogram(train_total_utterance_lengths, "Total Utterance Length Distribution (Train Data)", "train_total_utterance_length_distribution", "Length", "Frequency", x_tick_step=50)

print("Dev Data Conversation Length Distribution:")
plot_histogram([len(entry['input']['conversation']) for entry in dev_data], "Conversation Length Distribution (Dev Data)", "dev_conversation_length_distribution", "Length", "Frequency")

print("Dev Data Total Utterance Length Distribution:")
plot_histogram(dev_total_utterance_lengths, "Total Utterance Length Distribution (Dev Data)", "dev_total_utterance_length_distribution", "Length", "Frequency", x_tick_step=50)

print("Trainable (Train+Dev) Data Conversation Length Distribution:")
plot_histogram([len(entry['input']['conversation']) for entry in trainable_data], "Conversation Length Distribution (Train+Dev Data)", "train+dev_conversation_length_distribution", "Length", "Frequency")

print("Trainable (Train+Dev) Data Total Utterance Length Distribution:")
plot_histogram(dev_total_utterance_lengths, "Total Utterance Length Distribution (Train+Dev Data)", "train+dev_total_utterance_length_distribution", "Length", "Frequency", x_tick_step=50)

print("Test Data Conversation Length Distribution:")
plot_histogram([len(entry['input']['conversation']) for entry in test_data], "Conversation Length Distribution (Test Data)", "test_conversation_length_distribution", "Length", "Frequency")

print("Test Data Total Utterance Length Distribution:")
plot_histogram(test_total_utterance_lengths, "Total Utterance Length Distribution (Test Data)", "test_total_utterance_length_distribution", "Length", "Frequency", x_tick_step=50)

# Save total_utterance_lengths to CSV
def save_utterance_lengths_to_csv(data, filename):
    os.makedirs(f'{EDA_RESULT_PATH}', exist_ok=True)
    with open(f'{EDA_RESULT_PATH}/{filename}.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "total_utterance_length"])
        for entry in tqdm(data, desc="Saving utterance lengths to CSV"):
            utterance_length = sum(len(str(utterance['utterance'])) for utterance in entry['input']['conversation'])
            writer.writerow([entry['id'], utterance_length])

print("Saving Train Data Utterance Lengths to CSV:")
save_utterance_lengths_to_csv(train_data, "train_total_utterance_lengths")

print("Saving Dev Data Utterance Lengths to CSV:")
save_utterance_lengths_to_csv(dev_data, "dev_total_utterance_lengths")

print("Saving Trainable (Train+Dev) Data Utterance Lengths to CSV:")
save_utterance_lengths_to_csv(trainable_data, "train+dev_total_utterance_lengths")

print("Saving Test Data Utterance Lengths to CSV:")
save_utterance_lengths_to_csv(test_data, "test_total_utterance_lengths")


def tokenizer(text):
    return text.split(" ")

spwords = set(STOPWORDS)

mask = np.array(Image.open("assets/mask.png")) # 이미지를 불어와 arrary 데이터로 변환
def generate_wordcloud(text_data, title, filename):
    text = " ".join(map(str, text_data))
    token_list = tokenizer(text)
    
    filtered_token_list = []
    for token in token_list:
        token = re.sub(r'[ㅋ]+', 'ㅋㅋ', token)
        token = re.sub(r'[ㅎ]+', 'ㅎㅎ', token)
        token = re.sub(r'[ㅇ]+', 'ㅇㅇ', token)
        token = re.sub(r'[ㄱ]+', 'ㄱㄱ', token)
        token = re.sub(r'[ㄴ]+', 'ㄴㄴ', token)
        token = re.sub(r'[ㄷ]+', 'ㄷㄷ', token)

        if len(token) <= 1:
            continue
        filtered_token_list.append(token)
        
    token_counter = Counter(filtered_token_list)
    top_tokens = dict(token_counter.most_common(30))
    print(top_tokens)
    


    wordcloud = WordCloud(
        width=3000, height=2000,
        random_state=42, 
        mask=mask, background_color='white', colormap='Set2', 
        collocations=False,
        prefer_horizontal = 1,
        font_path='NanumGothic.ttf'
    )
    
    wordcloud.generate_from_frequencies(token_counter)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=25)
    plt.axis("off")
    os.makedirs(f'{EDA_RESULT_PATH}', exist_ok=True)
    plt.savefig(f'{EDA_RESULT_PATH}/{filename}.png', dpi=300)
    plt.close()

print("Generating Train Data Word Cloud:")
train_texts = list(set([utterance['utterance'] for entry in train_data for utterance in entry['input']['conversation']]))
generate_wordcloud(train_texts, "Word Cloud (Train Data)", "train_wordcloud")

print("Generating Dev Data Word Cloud:")
dev_texts = list(set([utterance['utterance'] for entry in dev_data for utterance in entry['input']['conversation']]))
generate_wordcloud(dev_texts, "Word Cloud (Dev Data)", "dev_wordcloud")

print("Generating Trainable (Train+Dev) Data Word Cloud:")
trainable_texts = list(set([utterance['utterance'] for entry in trainable_data for utterance in entry['input']['conversation']]))
generate_wordcloud(trainable_texts, "Word Cloud (Train & Dev Data)", "train+dev_wordcloud")

print("Generating Test Data Word Cloud:")
test_texts = list(set([utterance['utterance'] for entry in test_data for utterance in entry['input']['conversation']]))
generate_wordcloud(test_texts, "Word Cloud (Test Data)", "test_wordcloud")