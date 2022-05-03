#%%
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
import stanza
from spacy.matcher import DependencyMatcher
#%%
import spacy
nlp_spacy = spacy.load("en_core_web_sm")
#%%
nlp_stanza = stanza.Pipeline('en', use_gpu=False)
#%%
stanza.download('en') # download English model
# Make sure you have downloaded the StanfordNLP English model and other essential tools using,
# stanfordnlp.download('en')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#%%
def get_aspect_descriptors_1(character_list, text):
	stop_words = set(stopwords.words('english'))
	text = text.lower()
	sentList = nltk.sent_tokenize(text)

	fcluster = []
	totalfeatureList = []
	finalcluster = []
	dic = {}

	for line in sentList:
		text_list = nltk.word_tokenize(line) # Splitting up into words
		taggedList = nltk.pos_tag(text_list) # Doing Part-of-Speech Tagging to each word

		wordsList = [w for w in text_list if not w in stop_words]
		taggedList = nltk.pos_tag(wordsList)

		doc = nlp_stanza(text) # Object of Stanford NLP Pipeleine
		
		# Getting the dependency relations betwwen the words
		dep_node = []
		for dep_edge in doc.sentences[0].dependencies:
			# [0] -> Head word, [2] -> dependant word, [1] -> type of relation
			dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])

		# Coverting it into appropriate format
		for i in range(0, len(dep_node)):
			if (int(dep_node[i][1]) != 0):
				dep_node[i][1] = text_list[(int(dep_node[i][1]) - 1)]

		featureList = []
		categories = []
		for i in taggedList:
			if(i[1]=='JJ' or i[1]=='NN' or i[1]=='JJR' or i[1]=='NNS' or i[1]=='RB'):

				featureList.append(list(i)) # For features for each sentence
				totalfeatureList.append(list(i)) # Stores the features of all the sentences in the text
				categories.append(i[0])

		for i in featureList:
			filist = []
			for j in dep_node:
				if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
					if(j[0]==i[0]):
						filist.append(j[1])
					else:
						filist.append(j[0])
			fcluster.append([i[0], filist])

			print([i[0], filist])
			
	for i in totalfeatureList:
		dic[i[0]] = i[1]
	
	for i in fcluster:
		if(dic[i[0]]=="NN"):
			finalcluster.append(i)
		
	return(finalcluster)

#%%
def get_aspect_descriptors_2(character_list, text):
	sentences = sent_tokenize(text)

	aspects = []
	for sentence in sentences:
		doc = nlp_spacy(sentence)
		descriptive_term = ''
		target = ''
		for token in doc:
			if (token.pos_ == 'NOUN' or token.pos_ == "PROPN") :
				target = token.text
			if token.pos_ == 'ADJ':
				prepend = ''
				for child in token.children:
					if child.pos_ != 'ADV':
						continue
					prepend += child.text + ' '
				descriptive_term = prepend + token.text
		print({'aspect': target, 'description': descriptive_term})
		aspects.append({'aspect': target, 'description': descriptive_term})

	return aspects

# %%
text_1 = "The water is dirty. Martin was horrible."
text_2 = """
There was once a Bald Man who sat down after work on a
hot summer’s day. A Fly came up and kept buzzing about
his bald pate, and stinging him from time to time. The Man
aimed a blow at his little enemy, but acks palm came on his
head instead; again the Fly tormented him, but this time
the Man was wiser and said:
'You will only injure yourself if you take notice of despicable enemies.'"""
text_1 = text_1.replace("\n\n", " ").replace("\n", " ")
text_2 = text_2.replace("\n\n", " ").replace("\n", " ")


print(get_aspect_descriptors_1(['water', 'martin'], text_1))
print(get_aspect_descriptors_2(['water', 'martin'], text_1))
print(get_aspect_descriptors_1(['man', 'fly'], text_2))
print(get_aspect_descriptors_2(['man', 'fly'], text_2))

# %%
