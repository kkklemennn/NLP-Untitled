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
