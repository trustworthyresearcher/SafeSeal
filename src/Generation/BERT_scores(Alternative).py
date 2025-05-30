import os
from bert_score import score as bert_score

# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "hf_your_token_here"
cache_dir = 'Your/Cache/Directory'

def calc_scores(original_sentence, substitute_sentences):
    try:
        # Compute BERTScore for each substitute sentence relative to the original
        references = [original_sentence] * len(substitute_sentences)
        P, R, F1 = bert_score(cands=substitute_sentences, refs=references, model_type="bert-base-uncased")
        return F1.tolist()  # Return F1 scores as they represent overall similarity
    except Exception as e:
        print("Error computing BERTScore:", e)
        return []