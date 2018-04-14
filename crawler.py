import sys
import importlib
import copy
import os
import re
import queue
from lxml import html
from urllib.parse import urlparse, urljoin, urldefrag
import requests
import newspaper
from collections import deque
from bs4 import BeautifulSoup

BAD_EXTENSIONS = ['.pdf', '.png', '.jpeg', '.js', '.doc', '.docx', '.gif']

B_KEYWORDS = ['govern', 'conservative', 'senat', 'politic', 'election', 'congress', 'republic', 'democra', 'liberal', 'nation']

def normalize_url(url_source, potential_url):
    # if relative, make absolute
    if potential_url[:4] != 'http':
        potential_url = urljoin(url_source, potential_url)
    if potential_url[:5] == 'https':
        potential_url = 'http' + potential_url[5:]
    # remove ending '/'
    if potential_url[-1] == '/':
        potential_url = potential_url[:-1]
    # remove fragments if present
    p_url, frag = urldefrag(potential_url)
    return p_url.lower()


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


def crawl(url, num_links, site_name, score, politics_flag):
    # count = 0
    # paper = newspaper.build(url, memoize_articles=False)
    # print(url)
    # print(paper.size())
    # for article in paper.articles:
    #     if politics_flag and "politics" not in article.url:
    #         continue
    #     if count>num_links:
    #         break
    #     count+=1
    #     count += parseText(article, site_name, score)
    par_url = urlparse(url)
    org_domain = '{uri.scheme}://{uri.netloc}/'.format(uri=par_url)
    url_dict = set()
    frontier = deque()
    added = 0
    frontier.append(url)
    url_dict.add(url)
    added += 1
    while (len(frontier) > 0) and (added < num_links):
        next_url = frontier.popleft()
        r = ""
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        try:
            r = requests.get(next_url, headers=headers, timeout=3)
        except:
            continue
        if 'html' not in r.headers['content-type']:
            continue
        soup = BeautifulSoup(r.text, "lxml")
        for link in soup.find_all('a'):
            potential_url = link.get('href')
            if potential_url:
                p_url = normalize_url(next_url, potential_url)
                bad_ending = False
                for ending in BAD_EXTENSIONS:
                    if p_url.endswith(ending):
                        bad_ending = True
                        break
                if bad_ending:
                    continue
                # check for politics
                if politics_flag:
                    if "politics" not in p_url:
                        continue
                elif site_name == "Breitbart":
                    if not any (keyword in p_url for keyword in B_KEYWORDS):
                        continue
                elif site_name == "Reason":
                    if "blog" not in p_url:
                        continue
                # TODO: come up with way of identifying articles for Drudge and Breitbart
                par_url = urlparse(p_url)
                domain = '{uri.scheme}://{uri.netloc}/'.format(uri=par_url)
                if domain.endswith(org_domain):
                    if p_url[:4] == 'http' and '\n' not in p_url:
                        if p_url not in url_dict:
                            frontier.append(p_url)
                            added += 1
                            url_dict.add(p_url)
                            print("{}\t{}\t{}".format(p_url, site_name, score))


if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    file_path = sys.argv[1]
    num_links = int(sys.argv[2])
    with open(file_path, 'r') as curr_file:
        urls = [line.rstrip('\n') for line in curr_file]
        for url in urls:
            url = url.split('\t')
            site_name = url[1]
            score = url[2]
            politics_flag = False
            if url[3] == "yes":
                politics_flag = True
            crawl(url[0], num_links, site_name, score, politics_flag)
