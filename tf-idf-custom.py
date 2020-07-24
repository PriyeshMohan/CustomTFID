from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import math
import operator
from sklearn.preprocessing import normalize
import numpy as np

review_corpus = [
     'this is the first document',
     'this document is the second document',
     'and this is the third one',
     'is this the first document',
]

class CustomTfidfVectorizer:

	corpus = list()
	vocabulary = list()

	def __init__(self, c): 
		self.corpus = c
		self.vocabulary = self.fit()

	# Method returns sorted list of unique words from a corpus
	def fit(self):    
	    unique_words = set() # at first we will initialize an empty set
	    # check if its list type or not
	    if isinstance(self.corpus, (list,)):
	        for row in self.corpus: # for each review in the dataset
	            for word in row.split(" "): # for each word in the review. #split method converts a string into list of words
	                if len(word) < 2:
	                    continue
	                unique_words.add(word)
	        unique_words = sorted(list(unique_words))
	        return unique_words
	    else:
	        print("you need to pass list of sentance")

	# transfor to sparse matrix with value idf    
	def transform(self):
	    rows = []
	    columns = []
	    values = []
	    if isinstance(self.corpus, (list,)):
	        for idx, row in enumerate(tqdm(self.corpus)): # for each document in the dataset
	            # it will return a dict type object where key is the word and values is its frequency, {word:frequency}
	            word_freq = dict(Counter(row.split()))
	            # for every unique word in the document
	            for word, freq in word_freq.items():  # for each unique word in the review.                
	                if len(word) < 2:
	                    continue
	                # we will check if its there in the vocabulary that we build in fit() function
	                # dict.get() function will return the values, if the key doesn't exits it will return -1
	                # if the word exists
	                if word in self.vocabulary:
	                    # we are storing the index of the document
	                    rows.append(idx)
	                    # we are storing the dimensions of the word
	                    columns.append(self.vocabulary.index(word))
	                    # we are storing the idf value of the word
	                    values.append(self.idf_value(word,self.corpus))
	        return csr_matrix((values, (rows,columns)), shape=(len(self.corpus),len(self.vocabulary)))
	    else:
	        print("you need to pass list of strings")


    # Get the number of documents with 'term'  
	def no_documents_with_term(self,term, corpus):
		count = 0
		for each_string in corpus:
			if term in each_string:
				count += 1
		return count 
	
	# idf value for a given term in a corpus	
	def idf_value(self,term,corpus):
		return (1+np.log((1+len(corpus))/(1+self.no_documents_with_term(term,corpus))))	

	# idf values for the vocab
	def idf_value_for_vocab(self):
		vocab_idf_values = list()
		for each_unique_word in self.vocabulary:
			vocab_idf_values.append(self.idf_value(each_unique_word,self.corpus))
		return vocab_idf_values

	def get_sparse_matrix(self):
		# trasnform it to sparse matrix
		transformed_vector = self.transform()
		# do L2 normalisation
		normalised_matrix = normalize(transformed_vector, norm='l2')
		return normalised_matrix


# ---------------------- TASK 1 ---------------------

# init the custom tf idf vecotrizer
custom_vectorizer = CustomTfidfVectorizer(review_corpus)
# print the vocabulary
print(custom_vectorizer.vocabulary)
# print idf values for vocab
print(custom_vectorizer.idf_value_for_vocab())

# print the first row of noramlised sparse matrix
print(custom_vectorizer.get_sparse_matrix()[0])

# ---------------------- **** ---------------------


