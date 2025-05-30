import os
# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "hf_your_token_here"
cache_dir = 'Your/Cache/Directory'
import spacy
import json
from nltk.tokenize import word_tokenize
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import argparse
import nltk
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger_eng')
import time


from utils import extract_entities_and_pos, preprocess_text, split_sentences, look_up, look_up_with_cache
from randomization import randomize
nlp = spacy.load("en_core_web_sm")

Randomized_results = []

def process_target(pair, index, tokenizer, lm_model, Top_K, Final_K, threshold):
    """
    Process each target word to generate replacements.
    """
    word = pair[1]
    try:
        # Find alternatives for the target word
        list_alternative = look_up_with_cache(pair[0], pair[1], index, tokenizer, lm_model, Top_K, Final_K, threshold)
        #print(list_alternative)
        if not list_alternative:
            return None  # Skip if no valid alternatives found

        # Extract alternatives
        alternatives = [alt[0] for alt in list_alternative]
        similarity = [alt[1] for alt in list_alternative]

        if alternatives and similarity:
            # Apply randomization to find the final word
            randomized_word = randomize(word, alternatives, similarity)
            print(f" Word: {word}")
            print(f" Alternatives: {alternatives}")
            print(f" Randomized: {randomized_word}")
            print("-------------------")
            # Save the result
            Randomized_results.append({
                "word": word,
                "alternatives": alternatives,
                "randomized_word": randomized_word
            })
            return randomized_word
    except ValueError:
        return None

def save_results(output_name):
    with open(f"{output_name}_Randomization.json", "w") as f:
        json.dump(Randomized_results, f, indent=4)

def process_sentence(sentence, tokenizer, lm_model, Top_K, Final_K, threshold):
    """
    Process a single sentence to find randomized words for eligible target words.
    """
    replacements = []  # List to hold replacement tuples (index, target, replacement)
    doc = nlp(sentence)
    sentence_target_pairs = extract_entities_and_pos(sentence)

    for sent, target, position in sentence_target_pairs:
        # Match spaCy tokens to ensure accurate alignment
        spacy_tokens = [token.text for token in doc]
        if position < len(spacy_tokens) and spacy_tokens[position] == target:
            replacement = process_target((sentence, target), position, tokenizer, lm_model, Top_K, Final_K, threshold)
            if replacement:
                replacements.append((position, target, replacement))

    return replacements


def apply_replacements(sentence, replacements):
    """
    Apply replacements to the sentence while preserving original formatting, spacing, and punctuation.
    """
    doc = nlp(sentence)  # Tokenize the sentence
    tokens = [token.text_with_ws for token in doc]  # Preserve original whitespace with tokens

    # Apply replacements based on token positions
    for position, target, replacement in replacements:
        if position < len(tokens) and tokens[position].strip() == target:
            tokens[position] = replacement + (" " if tokens[position].endswith(" ") else "")

    # Reassemble the sentence
    return "".join(tokens)


def process_text(text, tokenizer, lm_model, Top_K, Final_K, threshold):
    """
    Processes text to replace words while preserving the original format, including spaces and newlines.
    """
    lines = text.splitlines(keepends=True)  # Retain original newline characters
    final_text = []
    total_randomized_words = 0
    total_words = len(word_tokenize(text))

    for line in lines:
        if line.strip():  # Process non-empty lines
            replacements = []
            sentence_replacements = process_sentence(
                line.strip(), tokenizer, lm_model, Top_K, Final_K, threshold
            )
            if sentence_replacements:
                replacements.extend(sentence_replacements)

            # Apply replacements to the original line
            if replacements:
                randomized_line = apply_replacements(line, replacements)
                final_text.append(randomized_line)
                total_randomized_words += len(replacements)
            else:
                final_text.append(line)  # Keep the original line
        else:
            final_text.append(line)  # Preserve empty lines

    # Combine all lines while preserving formatting
    return "".join(final_text), total_randomized_words, total_words


def main(args):
    start_time_all = time.time()
    # Clear the cache
    torch.cuda.empty_cache()
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model)
    # Load the autodecoder model
    lm_model = RobertaForMaskedLM.from_pretrained(args.model, attn_implementation="eager")
    lm_model.eval()
    # Read json file
    N_start = 0
    N_end = 100
    generated_data = []
    input_counter = 0
    saving_freq = 1
    Top_K = args.top_k
    Final_K = args.final_k
    threshold = args.threshold
    output_name = f"Train_Llama3_top_{Final_K}_threshold_{threshold}_Uniform_{N_start}_{N_end}"

    # Load data from the specified JSON file
    with open(args.data, 'r') as f:
        data = json.load(f)
        data = [{"Input": item["input"], "Original_output": item["output_only"]} for item in data[N_start:N_end]]

    # Loop through the data
    for i, item in enumerate(data):
        start_time = time.time()
        print(f"Processing item {i+1} / {N_end - N_start}")
        text = item["Original_output"]
        query = item["Input"]
        final_text, total_randomized_words, total_words = process_text(text, tokenizer, lm_model, Top_K, Final_K, threshold)
        end_time = time.time()
        time_elapsed = end_time - start_time

        # Store the input and output in a dictionary
        data_dict = {
            "input": query,
            "Original_output": text,
            "Watermarked_output": final_text,
            "Total_randomized_words": total_randomized_words,
            "Total_words": total_words,
            "time": time_elapsed
        }
        save_results(output_name)
        # Append the dictionary to the list of generated data
        generated_data.append(data_dict)
        # Increment input counter
        input_counter += 1

        # Save the results freqently
        if input_counter % saving_freq == 0:
            # Check if the file exits
            if os.path.isfile(output_name + "_" + str(input_counter-saving_freq) + ".json"):
                os.remove(output_name + "_" + str(input_counter-saving_freq) + ".json")
            with open(output_name + "_" + str(input_counter) + ".json", "w") as json_file:
                json.dump(generated_data, json_file, indent=4)
    end_time_all = time.time()
    time_elapsed_all = end_time_all - start_time_all
    print(f"Total time elapsed: {time_elapsed_all} seconds")
    print(f"Total time per data: {time_elapsed_all / len(generated_data)} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Watermark Generation')
    parser.add_argument('--model', default='roberta-base', type=str, help='Model for Masked LM to produce alternatives')
    parser.add_argument('--top_k', default=15, type=int, help='Top potential alternatives that LM will produce (Not yet formatted and evaluated)')
    #### Recondmend to modify the below parameters
    parser.add_argument('--data', default='data/llama_test.json', type=str, help='Data in json format to apply watermarking')  
    parser.add_argument('--final_k', default=3, type=int, help='Final K alternatives to consider')
    parser.add_argument('--threshold', default=0.8, type=float, help='Threshold for similarity')
    main(parser.parse_args())