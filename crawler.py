import sys
import importlib
import copy
import os
import re
import queue
from lxml import html
import urllib.parse
import requests
import newspaper



def parseText(article, site_name, score):
    try:
        if site_name == "Breitbart":
            if "big-government" not in aritlce.url:
                print(article.url)
                return -1
        article.download()
        article.parse()
        return 0
    except:
        return -1


def crawl(url, num_links, output, site_name, score, politics_flag):
    count = 0
    paper = newspaper.build(url, memoize_articles=False)
    print(url)
    print(paper.size())
    for article in paper.articles:
        if politics_flag and "politics" not in article.url:
            continue
        if count>num_links:
            break
        count+=1
        count += parseText(article, site_name, score)

if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    file_path = sys.argv[1]
    num_links = int(sys.argv[2])
    with open('crawler.output', 'w') as output:
        with open(file_path, 'r') as curr_file:
            urls = [line.rstrip('\n') for line in curr_file]
            for url in urls:
                url = url.split('\t')
                site_name = url[1]
                score = url[2]
                politics_flag = False
                if url[3] == "yes":
                    politics_flag = True
                crawl(url[0], num_links, output, site_name, score, politics_flag)
