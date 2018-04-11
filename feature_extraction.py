import nltk
import sys
import os
import newspaper
from lxml import html
import requests
from newspaper import Article
from enum import Enum
from nltk.tokenize import word_tokenize, sent_tokenize

MEN = ["ryan", "bill clinton", "sanders", "sessions", "mueller", "mccain", "pence", "biden", "booker", "cohen"]
WOMEN = ["hillary clinton", "devos", "michelle obama", "pelosi", "conway", "ivanka", "palin", "walters", "warren", "fiorina", "swisher"]

class Gender(Enum):
	MALE = 1
	FEMALE = 2
	NONE = 3

def process_doc(url):
	try:
		article = Article(url)
		article.download()
		article.parse()
	except:
		return -1
	text = article.text
	if not text:
		return -1
	# text_tokenized = word_tokenize(text)
	sents = sent_tokenize(text.lower())
	feature_dict = {"title": article.title, "text": sents}
	feature_dict["gender"] = get_people(feature_dict["title"].lower())
	#print(feature_dict["title"], feature_dict["gender"])

def get_people(title):
	if "trump" in title and "ivanka"  not in title and "melania"  not in title:
		return Gender.MALE
	if "obama"  in title and "michelle"  not in title:
		return Gender.MALE
	if any (name in title for name in MEN):
		return Gender.MALE
	if any (name in title for name in WOMEN):
		return Gender.FEMALE
	return Gender.NONE
	
def main():
	file_path = sys.argv[1]
	with open(file_path, 'r') as curr_file:
		urls = [line.rstrip('\n') for line in curr_file]
		for u in urls:
			u = u.split('\t')
			url = u[0]
			site_name = u[1]
			score = u[2]
			process_doc(url)

if __name__ == '__main__':
	main()
