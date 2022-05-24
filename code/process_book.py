# %%
import argparse
from pathy import Path

parser = argparse.ArgumentParser(description='Build a knowledge base for a given text.')
parser.add_argument("--corpus", type=str, default="litbank")
parser.add_argument("--file", type=str, default="76_adventures_of_huckleberry_finn_brat.txt")
parser.add_argument("--visualize", type=str, default="n", required=False)

args = parser.parse_args()
corpus = args.corpus
file_name = args.file
visualize = args.visualize
visualize_flag = False

if corpus == "litbank":
	corpus_path = "../material/litbank_corpus/"
elif corpus == "medium":
	corpus_path = "../material/medium_stories_corpus/"
elif corpus == "short":
	corpus_path = "../material/short_stories_corpus/"
else:
	print("Unknown corpus.")
	exit()

if visualize == "y":
	visualize_flag = True

# =========== Read the book into a string ===========
def get_book_string(file_path):
	try:
		with open(file_path, 'r', encoding='utf8') as file:
			return file.read().rstrip()
	except FileNotFoundError:
		print("File was not found. Please check if the file exists or correct your path.\n")
		exit()


print("\nProcessing ", file_name)
novel = get_book_string(corpus_path + file_name)


# %%
print("Importing libraries...")
from tokenize import String
from matplotlib.pyplot import get
import nltk
import spacy
import pandas as pd
import stanza
import spacy_stanza
import itertools
import pandas as pd
import collections
from pyvis.network import Network

#%%
# Spacy wrapper for StanfordNLP ner tagger
print("Creating NLP pipelines..")
ner_spacy_stanza_tagger = spacy_stanza.load_pipeline('en', processors={'ner': 'conll02'}, use_gpu=False)
stanza_sentiment = stanza.Pipeline(lang='en', processors='tokenize,sentiment', use_gpu=False)
nlp = spacy.load('en_core_web_sm')

#%%
# ===================================== NER =====================================
def remove_multiple_spaces(token):
	return " ".join(token.split())

def clean_entity(token):
	toclean = ['\'s', 'the ', '--', '/', '.']
	for el in toclean:
		if el in token.lower():
			token = token.lower().replace(el, '')
	return remove_multiple_spaces(token)

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

def visualize_entities_spacy(novel):
   html = spacy.displacy.render(nlp(novel), style="ent", options={"ents": ['PERSON', 'ORG']}, page=True)
   output_path = Path("ner-visualization.html")
   output_path.open("w", encoding="utf-8").write(html)

# =====================================/ NER =====================================

# ===================================== Sentiment analysis =====================================
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
# =====================================/ Sentiment analysis =====================================

# ===================================== Co-occurence =====================================
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

def sentences_where_entities_mentioned(sentences, entities_split):
	new_sentences = []
	for sentence in sentences:
		entities_mentioned = [word for word in sentence if word.lower() in entities_split]
		if len(entities_mentioned)!=0:
			new_sentences.append(sentence)
	return new_sentences

##%%
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

#%%
# Return index of the occurence or -1 otherwise
def find_duplicate_occurence(occurences, char1, char2):
	for ix, occurence in enumerate(occurences):
		if char1 in occurence and char2 in occurence:
			return ix
	return -1

def match_entities_for_graph(entities, edges, sentiments):
	occurences = []

	for e in edges:
		src_found = -1
		dst_found = -1
		for ix, entity in enumerate(entities):
			for entity_split in entity.split(" "):
				if entity_split in e[0].lower(): #or src in entity_split:
					src_found = ix
				elif entity_split in e[1].lower(): #or dst in entity_split:
					dst_found = ix

		if src_found != -1 and dst_found != -1:
			# check if the pair is already registered
			ix = find_duplicate_occurence(occurences, entities[src_found], entities[dst_found])
			# if not append a new occurence
			# also get the sentiments
			if ix == -1:
				ent1 = entities[src_found]
				ent2 = entities[dst_found]
				occurences.append([ent1, ent2, sentiments[ent1]['avg_sentiment'], sentiments[ent2]['avg_sentiment'], e[2]])
			# otherwise ++ its occurence number
			else:
				occurences[ix][2] += e[2]



	print(occurences)

	return occurences

# Return a hex color code for intervals
# from -1 to 1
# (-1 = red = bad, 0 = white = neutral, +1 = green = good)
def interval_to_hex(number):
	if number > 0.7 :
		return "#33cc33"
	elif number > 0.5 :
		return "#5cd65c"
	elif number > 0.3 :
		return "#85e085"
	elif number > 0.2 :
		return "#adebad"
	elif number > 0.1 :
		return "#d6f5d6"
	elif number > -0.1:
		return "#ffffff"
	elif number > -0.2 :
		return "#ffcccc"
	elif number > -0.3 :
		return "#ff9999"
	elif number  -0.5 :
		return "#ff6666"
	elif number > -0.7 :
		return "#ff3333"

def create_network(entities, sentiments):
	net = Network(height="1000px", width="95%", bgcolor="#FFFFFF", font_color="black", notebook=True)
	got_data = pd.read_csv("output.csv")

	sources = got_data['first']  # count
	targets = got_data['second']  # first
	weights = got_data['count']  # second

	edges = list(zip(sources, targets, weights))
	network_list = match_entities_for_graph(entities, edges, sentiments)

	for pair in network_list:
		#title1 = "{name} (avg. sentiment = {sentiment})".format(name=pair[0], sentiment=round(pair[2], 3))
		#title2 = "{name} (avg. sentiment = {sentiment})".format(name=pair[1], sentiment=round(pair[3], 3))
		title1 = "{name} ({sentiment})".format(name=pair[0], sentiment=round(pair[2], 3))
		title2 = "{name} ({sentiment})".format(name=pair[1], sentiment=round(pair[3], 3))
		net.add_node(title1, title1, title=title1, color=interval_to_hex(pair[2]))
		net.add_node(title2, title2, title=title2, color=interval_to_hex(pair[3]))
		net.add_edge(title1, title2, value=interval_to_hex(pair[4]))
	
	# Add non linked entities as sole nodes
	for entity in entities:
		flag = False
		for entry in network_list:
			if entry[0] == entity or entry[1] == entity:
				flag = True
				break
		if not flag:
			sentiment = round(sentiments[entity]['avg_sentiment'],3)
			title1 = "{name} ({sentiment})".format(name=entity, sentiment=sentiment)
			net.add_node(title1, title=title1, color=interval_to_hex(sentiment))

	return net

def find_cooccurences(novel, entities, sentiments):
	# Preprocess sentences, get entities
	filename = "list.csv"
	colname = "sentences"
	sentences = nltk.sent_tokenize(novel)

	# Write all sentences to .csv
	list_df = pd.DataFrame(columns=[colname])
	for sentence in sentences:    
		sentence_df = pd.DataFrame([sentence], index=list_df.columns).T
		list_df = pd.concat([list_df, sentence_df])
	list_df.to_csv(filename, mode = 'w', encoding='utf_8_sig')


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

	#Temporarily save data
	ct = collections.Counter(target_combinations)
	pd.DataFrame([{'first' : i[0][0], 'second' : i[0][1], 'count' : i[1]} for i in ct.most_common()]).to_csv('output.csv', index=False, encoding="utf_8_sig")
	
	if len(entities)==1:
		entity = entities[0]
		sentiment = round(sentiments[entity]['avg_sentiment'],3)
		net = Network(height="1000px", width="95%", bgcolor="#FFFFFF", font_color="black", notebook=True)
		title1 = "{name} ({sentiment})".format(name=entity, sentiment=sentiment)
		net.add_node(title1, title=title1, color=interval_to_hex(sentiment))
		net.show("cooccurrence-network.html")
	else:	
		try:
			net = create_network(entities, sentiments)
			net.show("cooccurrence-network.html")
		except:
			print("Could not draw cooccurrence network!")
# =====================================/ Co-occurence =====================================



# ===================================== Driver code =====================================

print("=========== Detecting entities with Stanza... ===========\n")
named_entities = get_common_entities_stanza(novel)
print(named_entities)
print("\n=========== Performing sentiment analysis... ===========\n")
sentiments = get_sentiments_for_characters_naive(named_entities, novel, True)
print(sentiments)
print("\n=========== Resolving cooccurence... ===========\n")
find_cooccurences(novel, named_entities, sentiments)
print("You can view the visual representation of character co-occurence by viewing the cooccurrence-network.html file.\n")
print("\n=========== Visualizing NER with Spacy... ===========\n")
visualize_entities_spacy(novel)
print("You can view the visual representation of named entity recognition by viewing the ner-visualization.html\n\n")

if visualize_flag:
	import os
	try:
		os.system("start cooccurrence-network.html")
		os.system("start ner-visualization.html")
	except:
		print("Could not open visualizations automatically.")
# %%
