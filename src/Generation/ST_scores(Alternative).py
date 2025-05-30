import os
import torch
from sentence_transformers import SentenceTransformer, util

# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "hf_your_token_here"
cache_dir = 'Your/Cache/Directory'

# Load the Sentence Transformer model
# Load model and move to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir).to(device)

def calc_scores(original_sentence, substitute_sentences):
    try:
        # Encode sentences into embeddings
        original_embedding = model.encode(original_sentence, convert_to_tensor=True)
        substitute_embeddings = model.encode(substitute_sentences, convert_to_tensor=True)

        # Compute cosine similarity scores
        similarity_scores = util.cos_sim(original_embedding, substitute_embeddings)

        # Convert tensor to a list (Ensure it is always a list)
        scores_list = similarity_scores.squeeze().tolist()

        # Ensure scores_list is always a list
        if isinstance(scores_list, float):  # Case when a single float is returned
            scores_list = [scores_list]

        return scores_list
    except Exception as e:
        print("Error computing Sentence Transformer similarity:", e)
        return []
