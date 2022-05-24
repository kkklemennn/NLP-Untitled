from unicodedata import name
import nltk
import json
import os
import numpy as np
import stanza
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from afinn import Afinn

# StanfordNLP sentiment
stanza_sentiment = stanza.Pipeline(lang='en', processors='tokenize,sentiment', use_gpu=False)

def get_sentiments_for_characters_naive(character_list, text, normalize):
	"""
	Get sentiments for the character list in the text.
	Perform sentiment analysis on each sentence where characters
	from the character list occur.
	Finally, calculate the average sentiment for each character (+1 postive , 0 neutral, -1 negative)
	"""
	# Create empty sentiment dictionary
	sentiments = {'stats' : {'sentiments': []}}
	# Create a stanza document from the entire text
	doc = stanza_sentiment(text)

	# Split "2 word" characters
	character_list_modified = [words for segments in character_list for words in segments.split()]
	#print("character list ->", character_list)
	# Loop over all sentences (stanza does the sent tokenization for us)
	for sentence in doc.sentences:
		# Convert stanza to python object
		word_list = sentence.to_dict()
		sentence_sentiment = sentence.sentiment - 1
		sentiments['stats']['sentiments'].append(sentence_sentiment)
		# Check if any of our characters are part of the sentence
		for word_ in word_list:
			word = word_['text'].lower()
			if word in character_list_modified:
				# Find on which index the "substring" of character occured
				# and save the key as the original name
				index = [idx for idx, s in enumerate(character_list) if word in s][0]
				word = character_list[index]
				# Check if sentiment for the character already exists
				# and append to existing sentiments
				if word in sentiments:
					sentiments[word]['sentiments'].append(sentence_sentiment)		
				# Create new list otherwise
				else:
					sentiments[word] = {'sentiments': [sentence_sentiment]}
	
	# Perform analysis
	if len((sentiments['stats']['sentiments'])) == 0:
		return None
	#
	sentiments['stats']['avg_sentiment'] = sum(sentiments['stats']['sentiments']) / len((sentiments['stats']['sentiments']))
	sentiments['stats']["num_sentences"] = len(sentiments['stats']['sentiments'])
	
	for key in sentiments:
		if key != "stats":
			sentiments[key]['avg_sentiment'] = sum(sentiments[key]['sentiments']) / len((sentiments[key]['sentiments']))
			if normalize:
				sentiments[key]['avg_sentiment'] -= sentiments['stats']['avg_sentiment']
	return sentiments


def get_book_string(file_path):
	with open(file_path, 'r', encoding="utf8") as file:
		return file.read().replace('\n', ' ')

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
		sentiment_score = afinn.score(sentence) / 5
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
	return sentiment_dict

# =========== SENTIMENT USING AFINN ===========

def generate_sentiment_results_afinn(stories_path):
	current_model = stories_path.split("/")[-2].split("_")[0]
	print("GENERATING SENTIMENT ANALYSIS RESULTS FOR " + current_model.upper() + " STORIES WITH STANZA")
	model = "stanza"
	files = os.listdir(stories_path)
	sentiments = {}
	for index, file in enumerate(files):
		novel = get_book_string(stories_path + file)
		entities = get_entities(stories_path + file, model)
		sentiments[file] = get_sentiment(novel, entities)

	with open("../results/sentiments_stanza_" + current_model + "_afinn.json", "w+", encoding="utf-8") as outfile:
		json.dump(sentiments, outfile, indent=4, ensure_ascii=False)
	print("DONE GENERATING")


# =========== SENTIMENT USING STANZA ===========

def generate_sentiment_results_stanza(stories_path):
	current_model = stories_path.split("/")[-2].split("_")[0]
	print("GENERATING SENTIMENT ANALYSIS RESULTS FOR " + current_model.upper() + " STORIES WITH STANZA")
	model = "stanza"
	files = os.listdir(stories_path)
	sentiments = {}
	for index, file in enumerate(files):
		print(file)
		sentiments[file] = {}
		novel = get_book_string(stories_path + file)
		entities = get_entities(stories_path + file, model)
		sentiments_normalized = get_sentiments_for_characters_naive(entities, novel, True)
		sentiments[file]['general'] = sentiments_normalized['stats']['avg_sentiment']
		sentiment = {}
		for entity in entities:
			if entity in sentiments_normalized: sentiment[entity] = sentiments_normalized[entity]['avg_sentiment']
			else: sentiment[entity] = 0.0
		sentiments[file]['entities'] = sentiment
		# if index > 0: break

	with open("../results/sentiments_stanza_" + current_model + "_stanza.json", "w+", encoding="utf-8") as outfile:
		json.dump(sentiments, outfile, indent=4, ensure_ascii=False)
	print("DONE GENERATING")


if __name__ == "__main__":
	short_stories_path = "../material/short_stories_corpus/"
	medium_stories_path = "../material/medium_stories_corpus/"
	litbank_stories_path = "../material/litbank_corpus/"

	generate_sentiment_results_stanza(short_stories_path)
	generate_sentiment_results_stanza(medium_stories_path)
	generate_sentiment_results_stanza(litbank_stories_path)
	generate_sentiment_results_afinn(short_stories_path)
	generate_sentiment_results_afinn(medium_stories_path)
	generate_sentiment_results_afinn(litbank_stories_path)
