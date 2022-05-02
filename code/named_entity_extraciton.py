# %%
import nltk
import spacy
import pandas as pd
import numpy as np
import stanza
import spacy_stanza
import timeit
import json
from nltk.corpus import stopwords

#%%
stanza.download('en', processors={'ner': 'conll02'})
# Classic spacy ner tagger
ner = spacy.load("en_core_web_md", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
# StanfordNLP ner tagger
ner_tagger_stanza = stanza.Pipeline('en', processors={'ner': 'conll02'}, use_gpu=False)
# Spacy rapper for StanfordNLP ner tagger
ner_spacy_stanza_tagger = spacy_stanza.load_pipeline('en', processors={'ner': 'conll02'}, use_gpu=False)

#%%
def remove_multiple_spaces(token):
	return " ".join(token.split())

def clean_entity(token):
    toclean = ['\'s', 'the ']
    for el in toclean:
        if el in token.lower():
            token = token.lower().replace(el, '')
    return remove_multiple_spaces(token)

def get_entities_with_spacy(text):
	doc = ner(text)
	occurence_list = []

	for entry in doc.ents:
		if entry.label_ == 'PERSON' or entry.label_ == 'ORG':
			occurence_list.append(entry)
	
	# transform to lowercase
	occurence_list = [str(el).lower() for el in occurence_list]
	# Remove duplicates
	occurence_list = list(dict.fromkeys(occurence_list))

	return occurence_list

def get_common_entities_spacy(text):
	# Return list of the most common characters in the story 
	# split into sentences
	sentences = nltk.sent_tokenize(text)

	occurence_list_sp = []
	occurence_list_nltk = []
	for sentence in sentences:
		occurence_list_sp.extend(get_entities_with_spacy(sentence))

	# Remove duplicates
	occurence_list = list(dict.fromkeys(occurence_list_sp))
	oc = []
	for el in occurence_list:
		oc.append(clean_entity(el))
	return oc

def get_common_entities_stanza(text):
    doc = ner_spacy_stanza_tagger(text)
    #all_entities = doc.ents
    all_entities =  []
    # for token in doc:
    #     if token.ent_type_ == "PERSON":
    #         all_entities.append(token.text)
    #     print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)

    for index, token in enumerate(doc):
        # print(index, token.text, token.ent_type_)
        stopwords = ["and", "a"]
        if token.ent_type_ == "PERSON" and token.text.lower() not in stopwords:
            current_person = []
            current_person.append(token.text)
            idx = index + 1
            next = doc[idx]
            while next.ent_type_ == "PERSON" and next.text.lower() not in stopwords:
                current_person.append(next.text)
                idx += 1
                next = doc[idx]
            current_person = ' '.join([str(item) for item in current_person])
            current_person = clean_entity(current_person)

            # Check if substring is in list
            flag = False
            for el in all_entities:
                if current_person in el:
                    flag = True
            if not flag:
                all_entities.append(current_person)

    #print("Unprocessed -> ", all_entities)
    # Transform Span object to list, make lowercase
    filtered_list = [entity.lower() for entity in all_entities]
    # Unify occurences
    filtered_list = list(dict.fromkeys(filtered_list))
    #...

    return filtered_list

#%%
def get_book_string(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        return file.read().rstrip()
	

# %% 

short_stories_path = "../material/short_stories_corpus/"
medium_stories_path = "../material/medium_stories_corpus/"

file_name = "The Most Dangerous Game.txt"
# =========== Read the book into a string ===========
novel = get_book_string(medium_stories_path + file_name)
if (novel is None):
	print("File was not found. Please check if the file exists or correct your path.")
	exit()

print("file string ->", novel[0:50])
#named_entities = get_common_entities_spacy(novel)
# print(get_common_entities_stanza(novel))
#stops = set(stopwords.words('english'))
#print(stops)

# %%
# =========== PERFORMANCE ANALYSIS ===========
def get_book_string(file_path):
	with open(file_path, 'r', encoding='utf8') as file:
		return file.read().rstrip()

short_stories_path = "../material/short_stories_corpus/"

files = os.listdir(short_stories_path)


def get_performance(model):
    performance_model = {}
    for file in files:
        print(file)
        performance = {}
        novel = get_book_string(short_stories_path + file)
        start = timeit.default_timer()
        if model == 'stanza':
            entities = get_common_entities_stanza(novel)
        elif model == 'spacy':
            entities = get_common_entities_spacy(novel)
        else:
            print("Enter either stanza or spacy")
            exit()
        occurences = {}
        for entity in entities:
            occurences[entity] = novel.lower().count(entity)
        stop = timeit.default_timer()
        performance['entities'] = occurences
        performance['runtime'] = stop - start
        performance_model[file] = performance
        print(performance)
        # break
    return performance_model

performance_stanza = get_performance('stanza')
performance_spacy = get_performance('spacy')

with open("../results/performance_stanza.json", "w+", encoding="utf-8") as outfile:
    json.dump(performance_stanza, outfile, indent=4, ensure_ascii=False)

with open("../results/performance_spacy.json", "w+", encoding="utf-8") as outfile:
    json.dump(performance_spacy, outfile, indent=4, ensure_ascii=False)

# %%
