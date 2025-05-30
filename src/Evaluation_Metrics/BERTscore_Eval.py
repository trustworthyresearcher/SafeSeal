import os
from bert_score import score
import json
import argparse
import csv
import torch
import warnings
warnings.filterwarnings("ignore")

# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "hf_your_token_here"
cache_dir = ''

def main(args):
    start = 0
    # Clear the cache
    torch.cuda.empty_cache()

    # Load Candidate and Reference Files if they are from the same file.
    with open(args.data_can, 'r') as f:
        data_1 = json.load(f)[start:args.N]
        cands = [item["Watermarked_output"] for item in data_1]
        # randomized_words = [item["Total_randomized_words"] for item in data_1]
        # total_words = [item["Total_words"] for item in data_1]


    with open(args.data_ref, 'r') as f:
        data_2 = json.load(f)[start:args.N]
        refs = [item["Original_output"] for item in data_2]

    # Set saving frequency
    saving_freq = 10
    # Initialize input counter
    input_counter = 0
    # Loop through the output text and detect the watermark
    results = []
    # Loop through the data and calculate the BERTScore
    for i, item in enumerate(cands):
            num_tokens = len(item.split())
            print(f"Item number: {i}")
            
            if num_tokens >= 16: # Only consider items with at least 16 tokens for valid assessment
                P, R, F1 = score([cands[i]], [refs[i]], lang="en", verbose=True)
                scores = F1.mean().item()
                #results.append([i, scores, randomized_words[i], total_words[i]])
                results.append([i, scores])

            else:
               print(f"Skipping item number {i} due to insufficient tokens.")
            # Write the results to a CSV file
            # Increment input counter
            input_counter += 1

            # Save the results after processing every saving_freq inputs
            if input_counter % saving_freq == 0:
                # Check if the file exits
                if os.path.isfile(f"{args.Output_name}{start}_{input_counter-saving_freq}.csv"):
                    os.remove(f"{args.Output_name}{start}_{input_counter-saving_freq}.csv")
                
                with open(f'{args.Output_name}{start}_{input_counter}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["data_item", 'BERTScore']) #, "Total_randomized_words", "Total_words"])  # Write the header
                    writer.writerows(results)  # Write the data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate BERTScore')
    parser.add_argument('--data_can',default= '',type=str, help='candidate data in json format')
    parser.add_argument('--data_ref',default= '',type=str, help='reference data in json format')
    parser.add_argument('--N', default= 1000, type=int, help='Number of data items to process')
    parser.add_argument('--Output_name', default= "BERTScore_Llama2_", type=str, help='Name of the output file')
    main(parser.parse_args())