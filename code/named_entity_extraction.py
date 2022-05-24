# %%
import nltk
import spacy
import pandas as pd
import numpy as np
import stanza
import spacy_stanza
import json
from nltk.corpus import stopwords

#%%
stanza.download('en', processors={'ner': 'conll02'})
# Classic spacy ner tagger
ner = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
# StanfordNLP ner tagger
ner_tagger_stanza = stanza.Pipeline('en', processors={'ner': 'conll02'}, use_gpu=False)
# Spacy rapper for StanfordNLP ner tagger
ner_spacy_stanza_tagger = spacy_stanza.load_pipeline('en', processors={'ner': 'conll02'}, use_gpu=False)

#%%
def remove_multiple_spaces(token):
	return " ".join(token.split())

def clean_entity(token):
    toclean = ['\'s', 'the ', '--', '/', '.']
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
    # Return list of the most common characters in the story 
	# split into sentences
    doc = ner_spacy_stanza_tagger(text)
    all_entities =  []

    for index, token in enumerate(doc):
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

    # Transform Span object to list, make lowercase
    filtered_list = [entity.lower() for entity in all_entities]
    # Unify occurences
    filtered_list = list(dict.fromkeys(filtered_list))

    return filtered_list

#%%
def get_book_string(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        return file.read().rstrip()
	

# %% 

short_stories_path = "../material/short_stories_corpus/"
medium_stories_path = "../material/medium_stories_corpus/"
litbank_stories_path = "../material/litbank_corpus/"

file_name = "Belling the Cat.txt" #174_the_picture_of_dorian_gray_brat.txt
# =========== Read the book into a string ===========
novel = get_book_string(short_stories_path + file_name)
if (novel is None):
	print("File was not found. Please check if the file exists or correct your path.")
	exit()

#print("file string ->", novel[0:50])
named_entities = get_common_entities_spacy(novel)
print(named_entities)
print(get_common_entities_stanza(novel))
#stops = set(stopwords.words('english'))
#print(stops)

# %%
import spacy
from nltk.tokenize import sent_tokenize
import itertools
from pyvis.network import Network
import pandas as pd
import collections

# Preprocess sentences, get entities
nlp = spacy.load('en_core_web_sm') ##nlp = ner
filename = "list.csv"
colname = "sentences"
sentences = sent_tokenize(novel)
entities = get_common_entities_stanza(novel) # all lowercase!
# print(entities)

# Write all sentences to .csv
list_df = pd.DataFrame(columns=[colname])
for sentence in sentences:    
    sentence_df = pd.DataFrame([sentence], index=list_df.columns).T
    list_df = list_df.append(sentence_df)
list_df.to_csv(filename, mode = 'w', encoding='utf_8_sig')


def sentence_separator(path, colname):
    df = pd.read_csv(path, encoding="utf_8_sig")
    data = df[colname]
    sentence = []
    for d in data:
        try:
            total_ls, noun_ls, verm_ls = ginza(d)
            sentence.append(total_ls)
        except:
            pass
    return sentence

def ginza(word):
    doc = nlp(word)
    #Survey results
    total_ls = []
    Noun_ls = [chunk.text for chunk in doc.noun_chunks]
    Verm_ls = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    for n in Noun_ls:
        total_ls.append(n)
    for v in Verm_ls:
        total_ls.append(v)
    return total_ls, Noun_ls, Verm_ls
"""-------------------------------------"""

def sentences_where_entities_mentioned(sentences, entities_split):
    new_sentences = []
    for sentence in sentences:
        entities_mentioned = [word for word in sentence if word.lower() in entities_split]
        if len(entities_mentioned)!=0:
            new_sentences.append(sentence)
    return new_sentences

#Separate sentences
sentences = sentence_separator(filename, colname)

#Split entities for easier coccurence matching e.g Dorian Gray -> Dorian, Gray
entities_split = [entity.split(" ") for entity in entities]
entities_split = [j for sub in entities_split for j in sub]

# Filter sentences - take only where entities are mentioned
sentences = sentences_where_entities_mentioned(sentences, entities_split)

sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
sentence_combinations = [[tuple(sorted(words)) for words in sentence] for sentence in sentence_combinations]
target_combinations = []

for sentence in sentence_combinations:
    target_combinations.extend(sentence)

# Determine if word pair contains two entites
def pair_in_entities(entities, src, dst):
    src = src.lower()
    dst = dst.lower()
    src_found = 0
    dst_found = 0

    for entity in entities:
        entities_split = entity.split(" ")
        for entity_split in entities_split:
            if entity_split in src: #or src in entity_split:
                src_found = 1
            elif entity_split in dst: #or dst in entity_split:
                dst_found = 1
    if src_found and dst_found:
         return 1
    else:
        return 0

def create_network():
    net = Network(height="1000px", width="95%", bgcolor="#FFFFFF", font_color="black", notebook=True)
    got_data = pd.read_csv("output.csv")

    sources = got_data['first']  # count
    targets = got_data['second']  # first
    weights = got_data['count']  # second

    edges = zip(sources, targets, weights)
    for e in edges:
        src = e[0]
        dst = e[1]
        w = e[2]
        if pair_in_entities(entities,src,dst):
            print(src,dst)
            net.add_node(src, src, title=src)
            net.add_node(dst, dst, title=dst)
            net.add_edge(src, dst, value=w)

    return net

#Temporarily save data
ct = collections.Counter(target_combinations)
pd.DataFrame([{'first' : i[0][0], 'second' : i[0][1], 'count' : i[1]} for i in ct.most_common()]).to_csv('output.csv', index=False, encoding="utf_8_sig")

net = create_network()
net.show("network.html")
