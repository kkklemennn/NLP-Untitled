# %%
import nltk
import spacy
import pandas as pd
import numpy as np
import stanza
import spacy_stanza
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# StanfordNLP sentiment
stanza_sentiment = stanza.Pipeline(lang='en', processors='tokenize,sentiment', use_gpu=False)

#%%
def get_sentiments_for_characters_naive(character_list, text):
	"""
	Get sentiments for the character list in the text.
	Perform sentiment analysis on each sentence and append the result
	to each character that occurs in the sentence.
	Finally, calculate the average sentiment for each character (+1 postive , 0 neutral, -1 negative)
	"""
	# Create empty sentiment dictionary
	sentiments = {'stats' : {'sentiments': []}}
	# Create a stanza document from the entire text
	doc = stanza_sentiment(text)

	# Loop over all sentences (stanza does the sent tokenization for us)
	for sentence in doc.sentences:
		# Convert stanza to python object
		word_list = sentence.to_dict()
		# Check if any of our characters are part of the sentence
		for word_ in word_list:
			word = word_['text'].lower()
			if word in character_list:
				sentiments['stats']['sentiments'].append(sentence.sentiment - 1)
				# Check if sentiment for the character already exists
				# and append to existing sentiments
				if word in sentiments:
					sentiments[word]['sentiments'].append(sentence.sentiment - 1)		
				# Create new list otherwise
				else:
					sentiments[word] = {'sentiments': [sentence.sentiment - 1]}
	
	# Perform analysis
	for key in sentiments:
		sentiments[key]['avg_sentiment'] = sum(sentiments[key]['sentiments']) / len((sentiments[key]['sentiments']))

	sentiments['stats']["num_sentences"] = len(sentiments['stats']['sentiments'])

	return sentiments


def get_book_string(file_path):
	with open(file_path, 'r', encoding="utf8") as file:
		return file.read().replace('\n', ' ')
	

# %% 

short_stories_path = "../material/short_stories_corpus/"
medium_stories_path = "../material/medium_stories_corpus/"

file_name = "Hills Like White Elephants.txt"
# =========== Read the book into a string ===========
novel = get_book_string(medium_stories_path + file_name)
if (novel is None or novel == ''):
	print("File was not found. Please check if the file exists or correct your path.")
	exit()

entities = ['man', 'jig', 'waitress', 'woman']


sentiments = get_sentiments_for_characters_naive(['luka', 'jan', 'martin'], 'Luka is very friendly. Jan is very mean. Martin is ok.')
print("Number of sentences :", sentiments["stats"]["num_sentences"])
print("average Sentiment :", sentiments["stats"]["avg_sentiment"])
for key in sentiments:
	if key != 'stats':
		print("Stats for character >>",key, "<< ") 
		print("Average sentiment: ", sentiments[key]['avg_sentiment']) 


# %%
# =========== SENTIMENT USING AFINN ===========
from afinn import Afinn
import json
import os
short_stories_path = "../material/short_stories_corpus/"
medium_stories_path = "../material/medium_stories_corpus/"
litbank_stories_path = "../material/litbank_corpus/"

model = "stanza"

def read_json(path):
    f = open(path,"r",encoding='utf-8')
    data = json.load(f)
    return data

def get_entities(book, model):
	corpus = book.split('/')[-2].split('_')[0]
	data = read_json('../results/performance_' + model + "_" + corpus + '.json')
	entities = list(data[book.split('/')[-1]]['entities'].keys())
	return entities


def get_sentiment(novel, entities):
	afinn = Afinn()
	sent_text = nltk.sent_tokenize(novel)
	sentiment_dict = {}
	sentiment = {}
	general_sentiment = []
	for sentence in sent_text:
		sentiment_score = afinn.score(sentence)
		general_sentiment.append(sentiment_score)
		for entity in entities:
			if entity in sentence.lower():
				if entity not in sentiment: sentiment[entity] = []
				sentiment[entity].append(sentiment_score)
				# print(sentiment_score)
	for entity in entities:
		if entity in sentiment: sentiment[entity] = np.mean(sentiment[entity])
		else: sentiment[entity] = 0.0
	general_sentiment = np.mean(general_sentiment)
	sentiment_dict["general"] = general_sentiment
	sentiment_dict["entities"] = sentiment
	print(sentiment_dict)

files = os.listdir(short_stories_path)
for index, file in enumerate(files):
	novel = get_book_string(short_stories_path + file)
	entities = get_entities(short_stories_path + file, model)
	get_sentiment(novel, entities)
	if index > 10:
		break

# %%
