import json
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from tqdm import tqdm


#Data
ref_data = "Test_LLaMA_top_3_threshold_0.8_Uniform_0_1000_1000.json"
ref_column = "Original_output" #


cand_data = "Test_LLaMA_top_3_threshold_0.8_Uniform_0_1000_1000.json"
cand_column ="Watermarked_output"# 

output_name = "Test.csv" 
N = 1000

# Use SPACY
# Load spaCy's Named Entity Recognition (NER) model
nlp = spacy.load("en_core_web_sm")

# Function to extract named entities from text
def extract_named_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


# === Similarity Calculation ===
def compute_similarity(entity1, entity2):
    if entity1 == entity2:
        return 1.0, 1.0, 1.0

    lev_similarity = SequenceMatcher(None, entity1, entity2).ratio()

    vec1 = nlp(entity1).vector.reshape(1, -1)
    vec2 = nlp(entity2).vector.reshape(1, -1)

    if np.any(vec1) and np.any(vec2):
        cos_similarity = cosine_similarity(vec1, vec2)[0][0]
    else:
        cos_similarity = 0.0

    combined_similarity = (lev_similarity + cos_similarity) / 2
    return combined_similarity, lev_similarity, cos_similarity

# === Greedy Pairwise Matching ===
def greedy_pairwise_matching(ref_entities, cand_entities):
    matched_entities = []
    cand_entities_copy = cand_entities.copy()

    for ref_entity in ref_entities:
        best_match = None
        best_score = 0
        best_lev_similarity = 0
        best_cos_similarity = 0

        for cand_entity in cand_entities_copy:
            similarity_score, lev_similarity, cos_similarity = compute_similarity(ref_entity, cand_entity)

            if similarity_score > best_score:
                best_score = similarity_score
                best_match = cand_entity
                best_lev_similarity = lev_similarity
                best_cos_similarity = cos_similarity

        if best_match:
            matched_entities.append((ref_entity, best_match, best_score, best_lev_similarity, best_cos_similarity))
            cand_entities_copy.remove(best_match)
        else:
            matched_entities.append((ref_entity, "MISSING", 0, 0, 0))

    for cand_entity in cand_entities_copy:
        matched_entities.append(("NEW ENTITY", cand_entity, 0, 0, 0))

    return matched_entities

# === Load Data ===
with open(ref_data, "r", encoding="utf-8") as ref_file:
    reference_data = [entry[ref_column] for entry in json.load(ref_file)[:N]]

with open(cand_data, "r", encoding="utf-8") as cand_file:
    candidate_data = [entry[cand_column] for entry in json.load(cand_file)[:N]]

assert len(reference_data) == len(candidate_data), "Mismatch in data point count!"


# === Process Each Pair ===
results = []

for idx, (ref_text, cand_text) in enumerate(tqdm(zip(reference_data, candidate_data), total=len(reference_data))):
    ref_entities = extract_named_entities(ref_text)
    cand_entities = extract_named_entities(cand_text)

    if len(ref_entities) == 0 and len(cand_entities) == 0:
        continue

    matched_entities = greedy_pairwise_matching(ref_entities, cand_entities)

    # Compute similarity lists
    cosine_similarities = [match[4] for match in matched_entities if match[4] > 0]
    levenshtein_similarities = [match[3] for match in matched_entities if match[3] > 0]

    avg_cosine_similarity = np.mean(cosine_similarities) if cosine_similarities else 0.0
    avg_levenshtein_similarity = np.mean(levenshtein_similarities) if levenshtein_similarities else 0.0
    avg_similarity = (avg_cosine_similarity + avg_levenshtein_similarity) / 2

    # Calculate exact match count
    exact_match_pairs = sum(1 for match in matched_entities if match[0] == match[1])

    # Union count
    union_count = len(ref_entities) + len(cand_entities) - exact_match_pairs
    union_count = union_count if union_count > 0 else 1  # Avoid division by zero

    # Final score
    final_score = (avg_similarity / union_count) * max(len(ref_entities), len(cand_entities))

    results.append({
        "Index": idx,
        "Reference_Entity_Count": len(ref_entities),
        "Candidate_Entity_Count": len(cand_entities),
        "Reference_Entities": ref_entities,
        "Candidate _Entities": cand_entities,
        "Matched_Entities": matched_entities,
        "Exact_Match Pairs": exact_match_pairs,
        "Union_Count": union_count,
        "Average_Cosine_Similarity": avg_cosine_similarity,
        "Average_Levenshtein_Similarity": avg_levenshtein_similarity,
        "Average_Combined_Similarity_Score": avg_similarity,
        "Final_Score": final_score
    })

# === Save to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(output_name, index=False)

print(f'Average Final Score: {results_df["Final_Score"].mean()}')