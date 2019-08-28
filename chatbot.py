import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')


fil = open('corpus.txt', 'r', errors = 'ignore')
raw = fil.read()
raw = raw.lower() #convert all text into lowercase

#Run below only on first execution
#nltk.download('punkt')
#nltk.download('wordnet')

sent_tokens = nltk.sent_tokenize(raw)# create list of sentances
word_tokens = nltk.word_tokenize(raw)# create list of words


lemmer = nltk.stem.WordNetLemmatizer()#WordNet is a semantically-oriented dictionary of English included in NLTK.

def LemTokens(tokens):
	return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text)	:
	return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "hey", "greetings", "what's up")
GREETING_RESPONSES = ["Hi", "Hey", "*nods*", "Hi there", "Hello"]

def greeting(sentence):
	for word in sentence.split():
		if word.lower() in GREETING_INPUTS:
			return random.choice(GREETING_RESPONSES)


def response(user_response):
	bot_response= ''
	sent_tokens.append(user_response)

	TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')
	tfidf = TfidfVec.fit_transform(sent_tokens)
	vals = cosine_similarity(tfidf[-1], tfidf)
	idx = vals.argsort()[0][-2]
	flat = vals.flatten()
	flat.sort()
	req_tfidf = flat[-2]

	if(req_tfidf ==0):
		bot_response = bot_response + "I am sorry! I don't understand you"
		return bot_response
	else:
		bot_response = bot_response + sent_tokens[idx]
		return bot_response


flag = True
print("ChatBot: I will answer your queries about datascience. If you want to exit, type BYE")
while(flag ==True):
	user_response = input().lower()
	if(user_response!= 'bye'):
		if(user_response == 'thanks' or user_response == 'thank you'):
			flag = False
			print("Chatbot: You are welcome")
		else:
			if(greeting(user_response)!= None):
				print("Chatbot:" + greeting(user_response))
			else:
				print("Chatbot:", end="")
				print(response(user_response))
				sent_tokens.remove(user_response)
	else:
		flag = False
		print("Chatbot: Bye! See you later")