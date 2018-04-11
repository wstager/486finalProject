# -*- coding: utf-8 -*-

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

MALE_PRONOUN = ["he", "he'd", "he's", "hes", "him", "himself", "his"]
FEMALE_PRONOUN = ["her", "hers", "herself", "she", "she'd", "she'll", "she's", "shes"]

LIWC_DICT = {}
LIWC_CATG = set()

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
	#text = text.replace("\\", "")
	sents = sent_tokenize(text.lower())
	feature_dict = {"title": article.title, "text": sents}
	feature_dict["gender"] = get_people(feature_dict["title"].lower())
	#print(feature_dict["title"], feature_dict["gender"])
	process_sentences(feature_dict["text"])

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


def read_liwc():
	with open("LIWC.2015.all", 'r') as liwc:
		for line in liwc:
			(key,val) = line.strip("\n").split(" ,")
			LIWC_CATG.add(val)
			if key in LIWC_DICT:
				LIWC_DICT[key].append(val)
			else:
				LIWC_DICT[key] = [val]
	print("Done reading")

def process_sentences(sents):
	male_cat = {key: 0 for key in LIWC_CATG}
	female_cat = {key: 0 for key in LIWC_CATG}

	for sent in sents:
		tokens = word_tokenize(sent)
		LIWC_analysis(tokens, male_cat, female_cat)
		
def LIWC_analysis(tokens, male_cat, female_cat):
	gender = get_sentence_gender(tokens)

	if gender is Gender.NONE:
		return

	for w in tokens:
		if w in LIWC_DICT:
			for cat in LIWC_DICT[w]:

				if gender is Gender.MALE:
					male_cat[cat] += 1

				if gender is Gender.FEMALE:
					female_cat[cat] += 1


def get_sentence_gender(tokens):
	male = False
	female = False
	if any (word in tokens for word in MALE_PRONOUN):
		male = True
	if any (word in tokens for word in FEMALE_PRONOUN):
		female = True
	if male and female:
		return Gender.NONE
	if male:	
		return Gender.MALE
	if female:
		return Gender.FEMALE
	return Gender.NONE


def main():	
	file_path = sys.argv[1]
	read_liwc()
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



