#%%
import os
from matplotlib.pyplot import text
import nltk
from nltk.tokenize import sent_tokenize
""" nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("reuters")
nltk.download("gutenberg")
nltk.download("wordnet")
nltk.download("tagsets")
nltk.download('punkt')
nltk.download('stopwords') """
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.collocations import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from sklearn.feature_extraction.text import TfidfVectorizer
# %%
# ================ Creating new corpus for our data ================
directory = '../material/short_stories_corpus/'
corpus = PlaintextCorpusReader(directory, '.*')
# Show some file ids just to see
file_names = corpus.fileids()
#print(file_names[:10])
# ================/ Creating new corpus for our data ================
# %%
# ================ Some basic_analysis information ================

def basic_analysis(corpus):
	all_letters_per_word = 0
	all_words_per_sentence = 0
	for file_name in file_names:
		n_chars = len(corpus.raw(file_name))
		n_chars_no_newlines = len(corpus.raw(file_name).replace("\r\n",""))
		n_words = len(corpus.words(file_name))
		n_sentences = len(corpus.sents(file_name))
		#n_sentences = len(sent_tokenize(corpus.raw(file_name).replace("\r\n","")))

		average_letters_per_word = n_chars_no_newlines/float(n_words)
		all_letters_per_word += average_letters_per_word
		average_words_per_sentence = n_words/float(n_sentences)
		all_words_per_sentence += average_words_per_sentence
		
		print("%s | %d characters | %d words | %d sentences | %.2f letters per word | %.2f words per sentence" % (file_name, n_chars, n_words, n_sentences, average_letters_per_word, average_words_per_sentence))

	average_letters_per_word = all_letters_per_word/float(len(file_names))
	average_words_per_sentence = all_words_per_sentence/float(len(file_names))
	average_sentences_per_story = len(corpus.sents())/float(len(file_names))
	print("TOTAL | %d characters | %d words | %.2f sentences | %.2f letters per word | %.2f words per sentence | %.2f average sentences per story" % (len(corpus.raw()), len(corpus.words()), len(corpus.sents()), average_letters_per_word, average_words_per_sentence, average_sentences_per_story))


basic_analysis(corpus) #short
print()

directory = '../material/medium_stories_corpus/' #medium
corpus = PlaintextCorpusReader(directory, '.*')
file_names = corpus.fileids()
basic_analysis(corpus)
print()

directory = '../material/litbank_corpus/' #litbank
corpus = PlaintextCorpusReader(directory, '.*')
file_names = corpus.fileids()
basic_analysis(corpus)
print()


#%% 
# ================ Some n-gram and collocations ================
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# Bigrams
finder = BigramCollocationFinder.from_words(corpus.words())
finder.apply_freq_filter(5)

#print("\nBest 50 bigrams according to PMI:", finder.nbest(bigram_measures.pmi, 50))
 
# Trigrams
finder = TrigramCollocationFinder.from_words(corpus.words())
finder.apply_freq_filter(5)
 
#print("\nBest 50 trigrams according to PMI:", finder.nbest(trigram_measures.pmi, 50))
# ================/ Some n-gram and collocations ================

# %%
# ================ Calculate TF-IDF scores for documents in the corpus ================
# Create a list of documents (containing actual text) in the corpus 
texts = []
for file_id in corpus.fileids():
	texts.append(corpus.raw(file_id))


vect = TfidfVectorizer() # parameters for tokenization, stopwords can be passed
tfidf = vect.fit_transform(texts)

#print("TF-IDF vectors (each column is a document):\n{}\nRows:\n{}".format(tfidf.T.A, vect.get_feature_names()))
# ================/ Calculate TF-IDF scores for documents in the corpus ================
