#%%
import os
from matplotlib.pyplot import text
import nltk
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
print(file_names[:10])
# ================/ Creating new corpus for our data ================
# %%
# ================ Some basic information ================
for file_name in file_names:
	n_chars = len(corpus.raw(file_name))
	n_words = len(corpus.words(file_name))
	n_sentences = len(corpus.sents(file_name))

	print("%s -> %d characters | %d words | %d sentences" % (file_name, n_chars, n_words, n_sentences))

print("TOTAL -> %d characters | %d words | %d sentences" % (len(corpus.raw()), len(corpus.words()), len(corpus.sents())))
# ================/ Some basic information ================
# %%

#%% 
# ================ Some n-gram and collocations ================
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# Bigrams
finder = BigramCollocationFinder.from_words(corpus.words())
finder.apply_freq_filter(5)

print("\nBest 50 bigrams according to PMI:", finder.nbest(bigram_measures.pmi, 50))
 
# Trigrams
finder = TrigramCollocationFinder.from_words(corpus.words())
finder.apply_freq_filter(5)
 
print("\nBest 50 trigrams according to PMI:", finder.nbest(trigram_measures.pmi, 50))
# ================/ Some n-gram and collocations ================

# %%
# ================ Calculate TF-IDF scores for documents in the corpus ================
# Create a list of documents (containing actual text) in the corpus 
texts = []
for file_id in corpus.fileids():
	texts.append(corpus.raw(file_id))


vect = TfidfVectorizer() # parameters for tokenization, stopwords can be passed
tfidf = vect.fit_transform(texts)

print("TF-IDF vectors (each column is a document):\n{}\nRows:\n{}".format(tfidf.T.A, vect.get_feature_names()))
# ================/ Calculate TF-IDF scores for documents in the corpus ================


# %%
