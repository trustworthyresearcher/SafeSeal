import os
from bleurt import score as bleurt_score

# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "hf_your_token_here"
# Set the cache directory for BLEURT    
cache_dir = 'Your/Cache/Directory'

# Initialize BLEURT scorer (adjust checkpoint path to your downloaded BLEURT model)
bleurt_scorer = bleurt_score.BleurtScorer("/Your_Directory_to_save/BLEURT-20")

def calc_scores(original_sentence, substitute_sentences):
    try:
        # Compute BLEURT scores for each substitute sentence relative to the original
        references = [original_sentence] * len(substitute_sentences)
        bleurt_scores = bleurt_scorer.score(references=references, candidates=substitute_sentences)
        return bleurt_scores
    except Exception as e:
        print("Error computing BLEURT scores:", e)
        return []

