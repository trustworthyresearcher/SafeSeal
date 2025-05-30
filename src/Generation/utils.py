import os
# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "hf_your_token_here"
# cache_dir = 'Your/Cache/Directory'
cache_dir = 'your_cache_directory_here'  # Replace with your actual cache directory

CACHE_FILE = "lookup_cache_llama2.json"


import re
import time
import math
import torch
import string
import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import json
from filelock import FileLock


from BLUERT_scores import calc_scores


lemmatizer = WordNetLemmatizer()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Define the detailed whitelist of POS tags (excluding adverbs)
DETAILED_POS_WHITELIST = {
    #'MD',  # Modal (e.g., can, could, will)
    'NN',  # Noun, singular or mass (e.g., dog, car)
    'NNS', # Noun, plural (e.g., dogs, cars)
    #'UH',  # Interjection (e.g., oh, wow)
    'VB',  # Verb, base form (e.g., run, eat)
    'VBD', # Verb, past tense (e.g., ran, ate)
    'VBG', # Verb, gerund or present participle (e.g., running, eating)
    'VBN', # Verb, past participle (e.g., run, eaten)
    'VBP', # Verb, non-3rd person singular present (e.g., run, eat)
    'VBZ', # Verb, 3rd person singular present (e.g., runs, eats)
    #'RP',  # Particle (e.g., up, off)
    'JJ',  # Adjective (e.g., big, blue)
    'JJR', # Adjective, comparative (e.g., bigger, bluer)
    'JJS'  # Adjective, superlative (e.g., biggest, bluest)
    'RB',  # Adverb (e.g., very, silently)
    'RBR', # Adverb, comparative (e.g., better)
    'RBS'  # Adverb, superlative (e.g., best)
    }

def extract_entities_and_pos(text):
    """
    Detect eligible tokens for replacement while skipping:
    - Named entities (e.g., names, locations, organizations).
    - Compound words (e.g., "Opteron-based").
    - Phrasal verbs (e.g., "make up", "focus on").
    - Punctuation and non-POS-whitelisted tokens.
    """
    doc = nlp(text)
    #entities = set(ent.text for ent in doc.ents)  # Extract named entities
    sentence_target_pairs = []  # List to hold (sentence, target word, token index)

    for sent in doc.sents:
        for token in sent:
            # Skip named entities using token.ent_type_ (more reliable than a text match)
            if token.ent_type_:
                continue

            # Skip standalone punctuation
            if token.is_punct:
                continue

            # Skip compound words (e.g., "Opteron-based")
            if "-" in token.text or token.dep_ in {"compound", "amod"}:
                continue

            # Skip phrasal verbs (e.g., "make up", "focus on")
            if token.pos_ == "VERB" and any(child.dep_ == "prt" for child in token.children):
                continue

            # Include regular tokens matching the POS whitelist
            if token.tag_ in DETAILED_POS_WHITELIST:
                sentence_target_pairs.append((sent.text, token.text, token.i))

    return sentence_target_pairs


def preprocess_text(text):
    """
    Preprocesses the text to handle abbreviations, titles, and edge cases
    where a period or other punctuation does not signify a sentence end.
    Ensures figures, acronyms, and short names are left untouched.
    """
    # Protect common abbreviations like "U.S." and "Corp."
    text = re.sub(r'\b(U\.S|U\.K|Corp|Inc|Ltd)\.', r'\1<PERIOD>', text)
    
    # Protect floating-point numbers or ranges like "3.57" or "1.48–2.10"
    text = re.sub(r'(\b\d+)\.(\d+)', r'\1<PERIOD>\2', text)
    
    # Avoid modifying standalone single-letter initials in names (e.g., "J. Smith")
    text = re.sub(r'\b([A-Z])\.(?=\s[A-Z])', r'\1<PERIOD>', text)

    # Protect acronym-like patterns with dots, such as "F.B.I."
    text = re.sub(r'\b([A-Z]\.){2,}[A-Z]\.', lambda match: match.group(0).replace('.', '<PERIOD>'), text)

    return text

def split_sentences(text):
    """
    Splits text into sentences while preserving original newlines exactly.
    - Protects abbreviations, acronyms, and floating-point numbers.
    - Only adds newlines where necessary without duplicating them.
    """
    # Step 1: Protect abbreviations, floating numbers, acronyms
    text = re.sub(r'\b(U\.S\.|U\.K\.|Inc\.|Ltd\.|Corp\.|e\.g\.|i\.e\.|etc\.)\b', r'\1<ABBR>', text)
    text = re.sub(r'(\b\d+)\.(\d+)', r'\1<FLOAT>\2', text)
    text = re.sub(r'\b([A-Z]\.){2,}[A-Z]\.', lambda m: m.group(0).replace('.', '<ABBR>'), text)

    # Step 2: Identify sentence boundaries without duplicating newlines
    sentences = []
    for line in text.splitlines(keepends=True):  # Retain original newlines
        # Split only if punctuation marks end a sentence
        split_line = re.split(r'(?<=[.!?])\s+', line.strip())
        sentences.extend([segment + "\n" if line.endswith("\n") else segment for segment in split_line])

    # Step 3: Restore protected patterns
    return [sent.replace('<ABBR>', '.').replace('<FLOAT>', '.') for sent in sentences]


def get_antonyms(word):
    '''
    Gets antonyms of a word in all its meanings from the WordNet knowledge base.
    * antonyms = words with an opposite meaning of the target word (day-night)
    '''
    ants = list()

    #Get antonyms from WordNet for this word and any of its synonyms.
    for ss in wn.synsets(word):
        ants.extend([lm.antonyms()[0].name() for lm in ss.lemmas() if lm.antonyms()]) 

    #Get snyonyms of antonyms found in the previous step, thus expanding the list even more.
    syns = list()
    for word in ants:
        for ss in wn.synsets(word):
            syns.extend([lm.name() for lm in ss.lemmas()])

    return sorted(list(set(syns)))


def get_pertainyms(word):
    '''
    Gets pertainyms of the target word from the WordNet knowledge base.
    * pertainyms = words pertaining to the target word (industrial -> pertainym is "industry")
    '''
    perts = list()
    for ss in wn.synsets(word):
        perts.extend([lm.pertainyms()[0].name() for lm in ss.lemmas() if lm.pertainyms()]) 
    return sorted(list(set(perts)))


def get_related_forms(word):
    '''
    Gets derivationally related forms (e.g. begin -> 'beginner', 'beginning')
    '''
    forms = list()
    for ss in wn.synsets(word):
        forms.extend([lm.derivationally_related_forms()[0].name() for lm in ss.lemmas() if lm.derivationally_related_forms()]) 
    return sorted(list(set(forms)))


def get_nyms(word, depth=-1):
    nym_list = ['antonyms', 'hypernyms', 'hyponyms', 'holonyms', 'meronyms', 
                'pertainyms', 'derivationally_related_forms']
    results = list()
    word = lemmatizer.lemmatize(word)

    def query_wordnet(getter):
        res = list()
        for ss in wn.synsets(word):
            res_list = [item.lemmas() for item in ss.closure(getter, depth=depth)]
            res_list = [item.name() for sublist in res_list for item in sublist]
            res.extend(res_list)
        return res

    for nym in nym_list:
        if nym=='antonyms':
            results.append(get_antonyms(word))

        elif nym in ['hypernyms', 'hyponyms']:
            getter = eval("lambda s : s."+nym+"()") 
            results.append(query_wordnet(getter))

        elif nym in ['holonyms', 'meronyms']:
            res = list()
            #Three different types of holonyms and meronyms as defined in WordNet
            for prefix in ['part_', 'member_', 'substance_']:
                getter = eval("lambda s : s."+prefix+nym+"()")
                res.extend(query_wordnet(getter))
            results.append(res)

        elif nym=='pertainyms':
            results.append(get_pertainyms(word))

        else:
            results.append(get_related_forms(word))

    results = map(set, results)
    nyms = dict(zip(nym_list, results))
    return nyms


#Converts a part-of-speech tag returned by NLTK to a POS tag from WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

#Function for clearing up duplicate words (capitalized, upper-case, etc.), stop words, and antonyms from the list of candidates.
def filter_words(target, words, scr, tkn):
    dels = list()
    toks = tkn.tolist()
    nyms = get_nyms(target)
    lemmatizer = WordNetLemmatizer()

    for w in words:
        if w.lower() in words and w.capitalize() in words:
            dels.append(w.capitalize())
        if w.lower() in words and w.upper() in words:
            dels.append(w.upper())
        if w in nltk.corpus.stopwords.words('english') or w in string.punctuation:
            dels.append(w)
        if lemmatizer.lemmatize(w.lower()) in nyms['antonyms']:
            dels.append(w)

    dels = list(set(dels))
    for d in dels:
        del scr[words.index(d)]
        del toks[words.index(d)]
        words.remove(d)

    return words, scr, torch.tensor(toks)

def is_valid_word(word):
    """
    Check if a word is valid using WordNet.
    """
    return bool(wn.synsets(word)) 

### LOOK-UP FUNCTION ###

def look_up(sentence, target, index, tokenizer, lm_model, Top_K, Final_K, threshold):
    """
    Finds replacement candidates for the target word in a preprocessed sentence.
    Ensures candidates meet the POS alignment, have no duplicates or antonyms,
    and handle multiple occurrences of the same word.
    """
    print(f' SENTENCE:  {sentence}')
    print(f' TARGET:  {target}')
    print(f' INDEX:  {index}')

    # Ensure input is preprocessed
    assert isinstance(sentence, str) and len(sentence) > 0, "Input sentence must be a non-empty string."

    # Tokenize sentence using spaCy for consistent tokenization
    doc = nlp(sentence)
    tokens = [token.text for token in doc]

    # Validate the index
    if index < 0 or index >= len(tokens):
        raise IndexError(f"Index {index} out of range for tokens: {tokens}")

    # Mask the target word at the specific index
    masked_tokens = tokens.copy()
    masked_tokens[index] = tokenizer.mask_token
    masked_sent = " ".join(masked_tokens)
    instruction = (
    "Given the context, replace the masked word with a word that fits grammatically, preserves the original meaning, and ensures natural flow in the sentence:")

    # Concatenate the original sentence and masked sentence
    input_text = f"{instruction} {sentence} {tokenizer.sep_token} {masked_sent}"

    # Get input token IDs for the concatenated input
    MAX_LENGTH = 512  # or 514, depending on your model

    input_ids = tokenizer.encode(input_text, add_special_tokens=True)

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]


    masked_position = input_ids.index(tokenizer.mask_token_id)

    # Get predictions from the model
    with torch.no_grad():
        output = lm_model(torch.tensor(input_ids).reshape(1, len(input_ids)))

    logits = output[0].squeeze()

    # Get top guesses: token IDs, scores, and words
    mask_logits = logits[masked_position].squeeze()
    top_tokens = torch.topk(mask_logits, k=Top_K, dim=0)[1]
    scores = torch.softmax(mask_logits, dim=0)[top_tokens].tolist()
    words = [tokenizer.decode(i.item()).strip() for i in top_tokens]

    # Filter initial words based on stopwords, punctuation, or target duplication
    words, scores, top_tokens = filter_words(target, words, scores, top_tokens)

    if len(words) == 0:
        return None

    # Sort words by scores in descending order
    sorted_words_scores = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
    sorted_words = [word for word, _ in sorted_words_scores]
    print(f' TOP GUESSES:  {sorted_words}')

    # Get POS tags for the original sentence
    original_pos = nltk.pos_tag(tokens)
    target_tag = original_pos[index][1]

    # Define the POS whitelist for filtering candidates
    DETAILED_POS_WHITELIST = {
    #'MD',  # Modal (e.g., can, could, will)
    'NN',  # Noun, singular or mass (e.g., dog, car)
    'NNS', # Noun, plural (e.g., dogs, cars)
    #'UH',  # Interjection (e.g., oh, wow)
    'VB',  # Verb, base form (e.g., run, eat)
    'VBD', # Verb, past tense (e.g., ran, ate)
    'VBG', # Verb, gerund or present participle (e.g., running, eating)
    'VBN', # Verb, past participle (e.g., run, eaten)
    'VBP', # Verb, non-3rd person singular present (e.g., run, eat)
    'VBZ', # Verb, 3rd person singular present (e.g., runs, eats)
    #'RP',  # Particle (e.g., up, off)
    'JJ',  # Adjective (e.g., big, blue)
    'JJR', # Adjective, comparative (e.g., bigger, bluer)
    'JJS'  # Adjective, superlative (e.g., biggest, bluest)
    'RB',  # Adverb (e.g., very, silently)
    'RBR', # Adverb, comparative (e.g., better)
    'RBS'  # Adverb, superlative (e.g., best)
    }

    # Antonym detection using WordNet
    target_synsets = wn.synsets(target)
    antonyms = {ant.name().split('.')[0] for syn in target_synsets for lem in syn.lemmas() for ant in lem.antonyms()}

    # Filter candidates for POS alignment and antonym removal
    filtered_words = []
    seen_words = set()

    for cand in sorted_words:
        # Skip duplicates
        cand_lower = cand.lower()
        if cand_lower in seen_words:
            continue
        seen_words.add(cand_lower)

        # Remove the original target word
        if cand_lower == target.lower():
            continue

        # Remove incomplete words
        if not is_valid_word(cand):
            continue

        # Get POS tags for the target word (initialize these before any checks)
        target_nltk_pos = nltk.pos_tag([target])[0][1]  # POS tag using nltk
        target_spacy_pos = nlp(target)[0].pos_  # POS tag using spaCy

        # Check if candidate and target have the same base form (using lemmatizer)
        target_lemma = lemmatizer.lemmatize(target)
        cand_lemma = lemmatizer.lemmatize(cand)
        if target_lemma != cand_lemma:
            # Get POS tags for candidate
            cand_nltk_pos = nltk.pos_tag([cand])[0][1]
            cand_spacy_pos = nlp(cand)[0].pos_

            # Compare POS tags; ensure both nltk and spaCy agree on the same tag
            if target_nltk_pos != cand_nltk_pos or target_spacy_pos != cand_spacy_pos:
                continue

        # Replace target with candidate and get new POS tags
        candidate_tokens = tokens.copy()
        candidate_tokens[index] = cand
        candidate_sentence = " ".join(candidate_tokens)
        new_nltk_pos = nltk.pos_tag(candidate_tokens)[index][1]
        new_spacy_pos = nlp(candidate_sentence)[index].pos_

        # Ensure the replacement has consistent POS alignment (nltk and spaCy must agree)
        if new_nltk_pos != target_nltk_pos or new_spacy_pos != target_spacy_pos:
            continue

        # Remove antonyms
        if cand in antonyms:
            continue

        # Add candidate if it passes all checks
        filtered_words.append(cand)

    # Prepare final candidate sentences
    candidate_sentences = [" ".join([cand if i == index else word for i, word in enumerate(tokens)])
                           for cand in filtered_words]
    final_scores = calc_scores(sentence, candidate_sentences)

    # Sort candidates by score
    sorted_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
    sorted_final_scores = [final_scores[i] for i in sorted_indices]
    sorted_filtered_words = [filtered_words[i] for i in sorted_indices]

    # Return all potential candidates (unfiltered)
    return list(zip(sorted_filtered_words, sorted_final_scores))

    # # Zip candidates and scores, keeping only top Final_K
    # final_results = list(zip(sorted_filtered_words, sorted_final_scores))[:Final_K]
    # print(f' FINAL RESULTS:  {final_results}')

    # # Only include results if there are exactly Final_K candidates and all meet the threshold
    # if len(final_results) == Final_K and all(score >= threshold for _, score in final_results):
    #     final_output = final_results
    #     print(f' FINAL OUTPUT:  {final_output}')  # Print before returning
    #     return final_output

    # # If conditions are not met, skip this instance
    # return None

### WORKING WITH CACHE ###
LOCK_FILE = f"{CACHE_FILE}.lock"

# Ensure the cache file exists
if not os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "w") as file:
        json.dump({}, file)

def load_cache():
    """Load the cache file with a lock."""
    with FileLock(LOCK_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)

def save_cache(cache):
    """Save the cache file with a lock."""
    with FileLock(LOCK_FILE):
        with open(CACHE_FILE, "w") as file:
            json.dump(cache, file, indent=4)

def look_up_with_cache(sentence, target, index, tokenizer, lm_model, Top_K=20, Final_K=3, threshold=0.75):
    """
    Caching-enabled version of the look_up function with dynamic filtering.
    """
    # Create a unique cache key for the context
    cache_key = f"{sentence}|{target}|{index}"

    # Load the cache
    cache = load_cache()

    # Check if the cache already has the result
    if cache_key in cache:
        cached_result = cache[cache_key]
        print(f"FOUND {cache_key}")

        # Retrieve potential results from cache
        sorted_final_scores = cached_result["sorted_final_scores"]
        sorted_filtered_words = cached_result["sorted_filtered_words"]

        # Apply filtering dynamically
        final_results = [
            (word, score) for word, score in zip(sorted_filtered_words, sorted_final_scores)
            if score >= threshold
        ][:Final_K]

        if len(final_results) == Final_K:
            print(f" FINAL RESULTS (from cache): {final_results}")
            return final_results
        else:
            return None

    # If not in cache, compute the results
    print(f"Cache miss for {cache_key}. Computing results...")
    result = look_up(sentence, target, index, tokenizer, lm_model, Top_K, Final_K, threshold)

    # Save all potential results to the cache
    if result:
        sorted_filtered_words, sorted_final_scores = zip(*result)
        cache[cache_key] = {
            "sentence": sentence,
            "target": target,
            "index": index,
            "sorted_final_scores": list(sorted_final_scores),
            "sorted_filtered_words": list(sorted_filtered_words)
        }
        save_cache(cache)

        # Apply filtering dynamically before returning
        final_results = [
            (word, score) for word, score in zip(sorted_filtered_words, sorted_final_scores)
            if score >= threshold
        ][:Final_K]

        print(f" FINAL RESULTS (from computation): {final_results}")
        if len(final_results) == Final_K:
            return final_results
        else:
            return None

    # If no results are found
    print("No valid candidates found.")
    return None