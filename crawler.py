import sys
import importlib
import copy
import os
import re
import Queue
from lxml import html
import urlparse
import requests


def crawl(url, num_links, output):
    visited = set()
    q = Queue.Queue()
    q.put(url)
    count = 0
    visited.add(url)
    #grabs domain from given url
    allowed_domain = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse.urlparse(url))
    with open('crawler.output.edges', 'w') as edge_output:
        while (not q.empty()) and (count < num_links):
            url = q.get()
            try:
                page = requests.get(url, timeout=3.0)
            except:
                #catches timeouts, any other possible errors
                continue
            if "html" in page.headers["Content-Type"]:
                count+=1
                output.write(url + '\n')
                tree = html.fromstring(page.text).iterlinks()
                for link in tree:
                    #ignore non-html files
                    if not any (ext in link[2] for ext in [".css", ".jpg", ".jpeg", ".docx", ".pdf", ".gif" ".ico", ".svg", ".ppt", ".js", ".png", ".cgi"]):
                        link = link[2]
                        #append relative URLs to parent URL
                        if link.startswith("/"):
                            link = urlparse.urljoin(allowed_domain,link)
                        #pulls domain from link to make sure it matches target domain
                        domain = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse.urlparse(link))
                        #remove trailing / for consistancy
                        link = link.rstrip('/')
                        #make sure domain matches original URL
                        if allowed_domain == domain:
                            if link == url:
                                continue
                            #add to queue if not visited
                            if link not in visited:
                                visited.add(link)
                                q.put(link)
                                edge_output.write(url + "\t" + link+"\n")
                            #store edges for pagerank
                            elif link != url:
                                edge_output.write(url + "\t" + link+"\n")

if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    file_path = sys.argv[1]
    num_links = int(sys.argv[2])
    with open('crawler.output', 'w') as output:
        with open(file_path, 'r') as curr_file:
            urls = [line.rstrip('\n') for line in curr_file]
            for url in urls:
                crawl(url, num_links, output)
