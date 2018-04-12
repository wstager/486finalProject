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
from nltk.sentiment.vader import SentimentIntensityAnalyzer

MEN = ["ryan", "bill clinton", "sanders", "sessions", "mueller", "mccain", "pence", "biden", "booker", "cohen"]
WOMEN = ["hillary clinton", "devos", "michelle obama", "pelosi", "conway", "ivanka", "palin", "walters", "warren", "fiorina", "swisher"]

MALE_PRONOUN = ["he", "he'd", "he's", "hes", "him", "himself", "his"]
FEMALE_PRONOUN = ["her", "hers", "herself", "she", "she'd", "she'll", "she's", "shes"]

LIWC_DICT = {}
LIWC_CATG = set()

SID = SentimentIntensityAnalyzer()

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
	#text = text.replace("\\", "")
	doc_sent = getSentiment(text)
	sents = sent_tokenize(text.lower())
	sentence_list_female = []
	sentence_list_male = []
	sentence_list_none = []
	female_sent = []
	male_sent = []
	for sent in sents:
		sentence_sent = getSentiment(sent)
		tokens = word_tokenize(sent)
		sentence_dict = {}
		for token in tokens:
			if token in sentence_dict:
				sentence_dict[token] += 1
			else:
				sentence_dict[token] = 1
		gender = get_sentence_gender(tokens)
		if gender == Gender.MALE:
			sentence_list_male.append(sentence_dict)
			male_sent.append(sentence_sent)
		elif gender == Gender.FEMALE:
			sentence_list_female.append(sentence_dict)
			female_sent.append(sentence_sent)
		else:
			sentence_list_none.append(sentence_dict)

	feature_dict = {"title": article.title, "text_male": sentence_list_male, "text_female": sentence_list_female, 
	                "text_none": sentence_list_none, "doc_sent": doc_sent, "female_sent": female_sent, "male_sent": male_sent}
	feature_dict["gender"] = get_people(feature_dict["title"].lower())
	
	male_LIWC = {key: 0 for key in LIWC_CATG}
	female_LIWC = {key: 0 for key in LIWC_CATG}
	none_LIWC = {key: 0 for key in LIWC_CATG}
	
	feature_dict["male_LIWC"] = male_LIWC
	feature_dict["female_LIWC"] = female_LIWC
	feature_dict["none_LIWC"] = none_LIWC

	feature_dict["male_ADJ"] = get_adjectives(sentence_list_male)
	feature_dict["female_ADJ"] = get_adjectives(sentence_list_female)

	return feature_dict

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

def getSentiment(text):
	ss = SID.polarity_scores(text)
	print(ss)
	return ss

def LIWC_helper(sentence_list, w, cat, star):

	for sentence in sentence_list:
		if star:
			for word in sentence:
				if word.startswith(w):
					for category in LIWC_DICT[w+"*"]:
						cat[category]+=sentence[word]
		else:
			if w in sentence:
				for category in LIWC_DICT[w]:
					cat[category] += sentence[w]

def LIWC_analysis(site_dict):

	for w in LIWC_DICT:
		star = False
		if w.endswith("*"):
			star = True
			w = w.rstrip("*")
		for site, s_dict in site_dict.items():
			for feature_dict in s_dict["doc_list"]:
				
				LIWC_helper(feature_dict["text_male"], w, feature_dict["male_LIWC"], star)
				LIWC_helper(feature_dict["text_female"], w, feature_dict["female_LIWC"], star)
				LIWC_helper(feature_dict["text_none"], w, feature_dict["none_LIWC"], star)

def get_adjectives(sentence_list):
	adjectives = {}
	for sentence in sentence_list:
		for w, count in sentence.items():
			pos = nltk.pos_tag([w])
			if "JJ" in pos[0][1]:
				if w in adjectives:
					adjectives[w] += count
				else:
					adjectives[w] = count
	return adjectives



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
	site_dict = {}
	with open(file_path, 'r') as curr_file:
		urls = [line.rstrip('\n') for line in curr_file]
		for u in urls:
			u = u.split('\t')
			url = u[0]
			site_name = u[1]
			score = u[2]
			feature_dict = process_doc(url)
			if site_name in site_dict:
				site_dict[site_name]["doc_list"].append(feature_dict)
			else:
				site_dict[site_name] = {"score": score, "doc_list": [feature_dict]}
	LIWC_analysis(site_dict)
	#print(site_dict)

if __name__ == '__main__':
	main()



