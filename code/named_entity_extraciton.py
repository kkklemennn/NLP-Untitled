# %%
import nltk
import spacy
import pandas as pd
import numpy as np
import stanza
import spacy_stanza
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
def get_entities_with_spacy(sentence):
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

	return occurence_list 

def get_common_entities_stanza(text):
	doc = ner_spacy_stanza_tagger(text)
	#all_entities = doc.ents
	all_entities =  []
	# for token in doc:
	# 	if token.ent_type_ == "PERSON":
	# 		all_entities.append(token.text)
	# 	print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)

	for index, token in enumerate(doc):
		if token.ent_type_ == "PERSON":
			current_person = []
			current_person.append(token.text)
			# print(index, token.text)
			idx = index + 1
			next = doc[idx]
			while next.ent_type_ == "PERSON":
				current_person.append(next.text)
				idx += 1
				next = doc[idx]
			current_person = ' '.join([str(item) for item in current_person])

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

## %% 
text = """The Fox and the Crow
A Fox once saw a Crow fly off with a piece of cheese in its
beak and settle on a branch of a tree. ‘That’s for me, as I
am a Fox,’ said Master Reynard, and he walked up to the
foot of the tree. ‘Good-day, Mistress Crow,’ he cried. ‘How
well you are looking to-day: how glossy your feathers; how
bright your eye. I feel sure your voice must surpass that of
other birds, just as your figure does; let me hear but one
song from you that I may greet you as the Queen of Birds.’
The Crow lifted up her head and began to caw her best, but
the moment she opened her mouth the piece of cheese fell
to the ground, only to be snapped up by Master Fox. ‘That
will do,’ said he. ‘That was all I wanted. In exchange for your
cheese I will give you a piece of advice for the future .’Do
not trust flatterers.’"""


#named_entities = get_common_entities_spacy(text)
#print(named_entities)

print(get_common_entities_stanza(text))
#stops = set(stopwords.words('english'))
#print(stops)

# %%
