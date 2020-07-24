import pickle
from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import math
import operator
from sklearn.preprocessing import normalize
import numpy as np



with open('cleaned_strings', 'rb') as f:
    review_corpus = pickle.load(f)
    
# printing the length of the corpus loaded
print("Number of documents in corpus = ",len(review_corpus))

class CustomLimitedVocabTfidfVectorizer:

	corpus = list()
	vocabulary = list()


	def __init__(self, c): 
		self.corpus = c
		self.vocabulary = self.fit()

	# Method returns sorted list of unique words from a corpus
	def fit(self):
	    unique_words = set()
	    idf_scores = list()
	    unique_trimmed_idf_values = list()
	    unique_trimmed_vocab_values = list()
	    # check if its list type or not
	    if isinstance(self.corpus, (list,)):
	        for row in self.corpus: # for each review in the dataset
	            for word in row.split(" "): # for each word in the review. #split method converts a string into list of words
	                if len(word) < 2:
	                    continue
	                unique_words.add(word)
	        # Get idf values for each word. idf values of words will be stored in their respective indexes
	        idf_scores = [self.idf_value(each_unique_word, self.corpus) for each_unique_word in unique_words]
	        # create a dict with key as word and values as idf value- since more than one word can have same idf value
	        vocab_dictionary = dict(zip(unique_words,idf_scores))
	        # sort the list descending idf scores
	        sorted_idf_score = sorted(idf_scores, reverse = True)
	        # trim the idf value to the first 50
	        unique_trimmed_idf_values = set(sorted_idf_score[:50])
	        # Loop through the first 50 idf value, get the saved value,ie, vocab word from the vocab_dictionary.
	        for each_idf_value in unique_trimmed_idf_values:
	        	list_of_list = list()
	        	# list_of_list will contain list of list of unique words having same idf value ie, [['very','is',..]] --> ['very','is' ..]
	        	list_of_list.append([key for key,value in vocab_dictionary.items() if value == each_idf_value])
	        	# flatten the list
	        	unique_trimmed_vocab_values.append([item for sublist in list_of_list for item in sublist])
	        # further flaten the unique_trimmed_vocab_values - since there will be a list for each idf values
	        top_vocab = [item for sublist in unique_trimmed_vocab_values for item in sublist]
	        # get top 50 vocab values
	        top_50_vocab = top_vocab[:50]
	        return top_50_vocab
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
		value = (1+np.log((1+len(corpus))/(1+self.no_documents_with_term(term,corpus))))
		return value

	# idf values for the vocab
	def idf_value_for_vocab(self):
		vocab_dictionary = dict()
		for each_unique_word in self.vocabulary:
			vocab_dictionary.update({each_unique_word: self.idf_value(each_unique_word,self.corpus)})
		return vocab_dictionary

	def get_sparse_matrix(self):
		# trasnform it to sparse matrix
		transformed_vector = self.transform()
		# do L2 normalisation
		normalised_matrix = normalize(transformed_vector, norm='l2')
		return normalised_matrix


# ---------------------- TASK 2 ---------------------

# init the custom tf idf vecotrizer
custom_limited_vocab_vectorizer = CustomLimitedVocabTfidfVectorizer(review_corpus)
# print the vocabulary
print(custom_limited_vocab_vectorizer.vocabulary)
# print idf values for vocab
print(custom_limited_vocab_vectorizer.idf_value_for_vocab())
# print normalised sparse matrix
print(custom_limited_vocab_vectorizer.get_sparse_matrix())
# 

# ---------------------- **** ---------------------