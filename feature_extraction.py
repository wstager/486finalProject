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
from copy import deepcopy
import math

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

"""
process the doc by downloading an article from the given url and then extracting features
INPUT: url is a link to each article our crawler found, 
		all_words is a dictionary of every word in each article. 
			Maps to another dictionary where the key is the gender associated with the overall sentence the word was in 
			and value is the count of number of times the word was in the sentence.
		class_counts is a dictionary that maps a gender to the number of times a sentence is classified as that gender

OUTPUT: feature_dict - a dictionary containing the following features: a url and title of the article, a list of male sentences,
		a list of female sentences, and a list of sentences with no gender, the overall document's sentiment,
		the average sentiment score of female and then male sentences, the gender of any influential political leaders in the title of the article,
		LIWC counts associated with the sentences in the set of male, female and none,
		a list of all adjectives in the female sentences and then also the male sentences.
"""
def process_doc(url, all_words, class_counts):
	try:
		article = Article(url)
		article.download()
		article.parse()
	except:
		return -1
	text = article.text
	if not text:
		return -1
	doc_sent = getSentiment(text)
	sents = sent_tokenize(text.lower())
	sentence_list_female = []
	sentence_list_male = []
	sentence_list_none = []
	female_sent = 0
	male_sent = 0
	for sent in sents:
		sentence_sent = getSentiment(sent)
		tokens = word_tokenize(sent)
		sentence_dict = {}
		gender = get_sentence_gender(tokens)
		if gender != Gender.NONE:
			class_counts[gender]+=1
		for token in tokens:
			pos = nltk.pos_tag([token])
			if not "JJ" in pos[0][1]:
				continue
			if token in all_words and gender in all_words[token]:
				all_words[token][gender] += 1
			elif token in all_words:
				all_words[token][gender] = 1
			else:
				all_words[token] = {gender: 1}
			if token in sentence_dict:
				sentence_dict[token] += 1
			else:
				sentence_dict[token] = 1
		
		if gender == Gender.MALE:
			sentence_list_male.append(sentence_dict)
			male_sent += sentence_sent
		elif gender == Gender.FEMALE:
			sentence_list_female.append(sentence_dict)
			female_sent += sentence_sent
		else:
			sentence_list_none.append(sentence_dict)
	if not sentence_list_female and not sentence_list_male:
		return -1

	if sentence_list_male:
		male_sent = float(male_sent) / float(len(sentence_list_male))
	if sentence_list_female:
		female_sent = float(female_sent) / float(len(sentence_list_female))

	feature_dict = {"url": url, "title": article.title, "text_male": sentence_list_male, "text_female": sentence_list_female, 
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

"""
INPUT: title, which is the title of a given article
OUTPUT: the gender of any influential political leaders we accounted for that was found in the title
"""
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

"""
read in the LIWC dictionary into a python dictionary where the key is the category and the value is a list of the associated words
"""
def read_liwc():
	with open("LIWC.2015.all", 'r') as liwc:
		for line in liwc:
			(key,val) = line.strip("\n").split(" ,")
			LIWC_CATG.add(val)
			if key in LIWC_DICT:
				LIWC_DICT[key].append(val)
			else:
				LIWC_DICT[key] = [val]

"""
INPUT: text, which is a sentence from an article
OUTPUT: the compound sentiment score of the given text
"""
def getSentiment(text):
	ss = SID.polarity_scores(text)
	return ss["compound"]

"""
INPUT: sentence_list is a list of dictionaries where each dictionary maps a token in the sentence to a count
		of how many times that word appears in the sentence
		w is the word we are analyzing in LIWC_analysis
		cat is a dictionary that maps all categories of a LIWC dict to a value, which is a count of how many words was associated with that category
		star is a boolean that indicates if a star was originally in the word (star means the word has multiple endings)
"""
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

"""
uses the LIWC dictionary to count the LIWC categories
INPUT: site_dict is a dictionary that maps a site to a feature dictionary
"""
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
"""
finds adjectives in a sentence using nltk part of speech and keeps a count of each adjective
INPUT: sentence_list is a list of dictionaries where each dictionary maps a token in the sentence to a count
OUTPUT: adjectives is a dictionary that maps all adjectives in a sentence to a count of how many times that adjective appeared in the sentence 
"""
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

"""
gets the gender of a sentence by looking at pronouns and names of influential political leaders
INPUT: tokens is a list of tokens in a sentence
OUTPUT: the gender of the given sentence
"""
def get_sentence_gender(tokens):
	male = False
	female = False
	if any (word in tokens for word in MALE_PRONOUN):
		male = True
	if any (word in tokens for word in FEMALE_PRONOUN):
		female = True
	name_gender = get_people(tokens)
	if name_gender == Gender.FEMALE:
		female = True
	elif name_gender == Gender.MALE:
		male = True
	if male and female:
		return Gender.NONE
	if male:	
		return Gender.MALE
	if female:
		return Gender.FEMALE
	return Gender.NONE

"""
analyzes all female and male adjectives and determines if they are meaningful by computing a probability
then prints the adjective to the female and male adjective files with it's associated values
INPUT: all_words is a dictionary of every word in each article
		threshold is a value that we wanted a count of an adjective to have in order to be considered meaningful
		f_adj_file is the name of the file where we store all the female adjectives
		m_adj file is the name of the file where we store all the male adjectives
		class_counts is a dicionary that maps a gender to a count of th enumber of sentences associated with that gender
"""
def word_analysis(all_words, threshold, f_adj_file, m_adj_file, class_counts):
	probs = {Gender.MALE: {}, Gender.FEMALE: {}}
	for word in all_words:
		total_occ = 0
		if Gender.MALE in all_words[word]:
			total_occ += all_words[word][Gender.MALE]
		if Gender.FEMALE in all_words[word]:
			total_occ += all_words[word][Gender.FEMALE]
		all_words[word]["total"] = total_occ
		if total_occ < threshold:
			continue
		if Gender.MALE in all_words[word]:
			probs[Gender.MALE][word] = float(all_words[word][Gender.MALE]/total_occ)
		if Gender.FEMALE in all_words[word]:
			probs[Gender.FEMALE][word] = float(all_words[word][Gender.FEMALE]/total_occ)
	normal_probs = deepcopy(probs)
	male_prob = float(class_counts[Gender.MALE])/float(class_counts[Gender.MALE] + class_counts[Gender.FEMALE])
	female_prob = float(class_counts[Gender.FEMALE])/float(class_counts[Gender.MALE] + class_counts[Gender.FEMALE])
	for word, prob  in normal_probs[Gender.MALE].items():
		normal_probs[Gender.MALE][word] = math.log(float(normal_probs[Gender.MALE][word])/float(male_prob))
	for word, prob  in normal_probs[Gender.FEMALE].items():
		normal_probs[Gender.FEMALE][word] = math.log(float(normal_probs[Gender.FEMALE][word])/float(female_prob))
	sorted_female = sorted(normal_probs[Gender.FEMALE].items(), key=lambda x:x[1], reverse=True)
	sorted_male = sorted(normal_probs[Gender.MALE].items(), key=lambda x:x[1], reverse=True)
	with open(f_adj_file, 'w+') as f_f:
		f_f.write("Word\tCond_Prob\tNorm_Prob\tCount\n")
		for word, value in sorted_female:
			word_count = all_words[word]["total"]
			prob = probs[Gender.FEMALE][word]
			f_f.write("{}\t{}\t{}\t{}\n".format(word, prob, value,word_count))
	with open(m_adj_file, 'w+') as f_m:
		f_m.write("Word\tCond_Prob\tNorm_Prob\tCount\n")
		for word, value in sorted_male:
			word_count = all_words[word]["total"]
			prob = probs[Gender.MALE][word]
			f_m.write("{}\t{}\t{}\t{}\n".format(word, prob, value, word_count))

"""
prints the site dict in a systematic way
INPUT: site_dict is a dictionary that maps a site to a feature dictionary
"""
def print_site_dict(site_dict):
	for site_name, s_dict in site_dict.items():
		with open("feat_files/feat_{}.txt".format(site_name), 'w+') as site_file:
			with open("adj_files/ADJ_{}.txt".format(site_name), 'w+') as adj_site_file:
				site_file.write("0_URL\t1_doc_gender\t2_doc_sent\t3_female_sent\t4_male_sent\t")
				count = 5
				sorted_LIWC = sorted(LIWC_CATG)
				for i in range(0,3):
					for key in LIWC_CATG:
						site_file.write(str(count)+"_"+str(i)+"_"+str(key)+"\t")
						count+=1
				site_file.write("\n")
				for feature_dict in s_dict["doc_list"]:
					site_file.write(feature_dict["url"]+"\t"+str(feature_dict["gender"])+"\t"+
						            str(feature_dict["doc_sent"])+"\t"+str(feature_dict["female_sent"])+"\t"+str(feature_dict["male_sent"])+"\t")
					for LIWC_key in sorted_LIWC:
						site_file.write(str(feature_dict["female_LIWC"][LIWC_key])+"\t")
					for LIWC_key in sorted_LIWC:
						site_file.write(str(feature_dict["male_LIWC"][LIWC_key])+"\t")
					for LIWC_key in sorted_LIWC:
						site_file.write(str(feature_dict["none_LIWC"][LIWC_key])+"\t")
					adj_site_file.write("male"+"\t"+feature_dict["url"]+"\n")
					for adj, count in feature_dict["male_ADJ"].items():
						adj_site_file.write(str(adj)+"\t"+str(count)+"\n")
					adj_site_file.write("female"+"\t"+feature_dict["url"]+"\n")
					for adj, count in feature_dict["female_ADJ"].items():
						adj_site_file.write(str(adj)+"\t"+str(count)+"\n")
					site_file.write("\n")
	
def main():	
	file_path = sys.argv[1]
	read_liwc()
	site_dict = {}
	all_words = {}
	class_counts = {Gender.MALE: 0, Gender.FEMALE: 0}
	with open(file_path, 'r') as curr_file:
		urls = [line.rstrip('\n') for line in curr_file]
		for u in urls:
			u = u.split('\t')
			url = u[0]
			site_name = u[1]
			score = u[2]
			feature_dict = process_doc(url, all_words, class_counts)
			if feature_dict == -1:
				continue
			if site_name in site_dict:
				site_dict[site_name]["doc_list"].append(feature_dict)
			else:
				site_dict[site_name] = {"score": score, "doc_list": [feature_dict]}
	LIWC_analysis(site_dict)
	word_analysis(all_words, 10, "adj_files/female_adj.txt", "adj_files/male_adj.txt", class_counts)
	print_site_dict(site_dict)
	

if __name__ == '__main__':
	main()



