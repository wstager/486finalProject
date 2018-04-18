PROJECT DEPENDENCIES
To run the project on CAEN, first make sure that Anaconda with Python version 3.6.4 is installed

To get the rest of the necessary libraries, open Python and type the following:
1. import nltk
2. nltk.download('averaged_perceptron_tagger')
3. nltk.download('vader_lexicon')
4. conda install -c conda-forge tenorflow
5. conda install -c conda-forge newspaper3k

PROJECT COMPONENTS

crawler.py
Description: Given a list of starting urls, crawls through each site to extract a given number politics-specific urls
Inputs: A list of starting urls, a maximum number of urls to crawl per site
Outputs: Prints extracted urls to standard output. Pipe them to a file to use them in future steps.
Command to run: python crawler.py <seed URL file> <number of URLS> > <crawler output file>
Example: python crawler.py myseedURLs.txt 1500 > crawler_output.txt

feature_extraction.py
Description: Given a list of urls to examine, computes feature values for each url. Also, produces the female_adj.txt,
male_adj.txt, ADJ_*.txt, and feat_*.txt files described below.
Inputs: A file with a list of urls to compute feature values for.
Outputs: Files containing the feature values for each site as well as files documenting the occurences of adjectives in all sites and specific to each article,
which are later used to identify informative adjectives.
Command to run: python feature_extraction.py <file with URLS>
Example: python feature_extraction.py crawler crawler_output.txt

nearulnet.py
Description: Trains and tests a neural network that predicts the political category (liberal, neutral, conservative) of newspaper articles from the feature values
computed at the feature extraction stage. Also analyzes the adjective files produces during feature extraction to determine a list of the most meaningful adjectives
and then compute counts of these adjectives for each doc to use as features. Evaluates the performance of the neural network by outputting accuracy, precision, and recall
statistics.
Inputs: Number of nodes to use in the hidden layer of the network. Adjective and feature files produced during the feature extraction stage.
Outputs: Accuracy, precision, and recall values for the neural network predictions.
Command to run: python neuralnet.py <number of nodes in hidden layer>
Example: python neuralnet.py 100

DATA FILES

crawler_output.txt: Contains a list of all urls collected during crawling.

adj_files/female_adj.txt: Contains a list of all adjectives occuring in a sentences identified as female that had more than 10 occurences.
Adjectives are listed as word, p(female | word), log(P(female | word)/ P(female)), count of number of times word occured in sentence with gender.

adj_files/male_adj.txt: Contains a list of all adjectives occuring in a sentences identified as male that had more than 10 occurences.
Adjectives are listed as word, p(male | word), log(P(male | word)/ P(female)), count of number of times word occured in sentence with gender.

adj_files/ADJ_<news site name>.txt: Contains a list of all article urls analyzed on the given <news site name>.
For each, a list of all adjectives in both male and female sentences and their counts is recorded.

feat_files/feat_<news site name>.txt: Contains a list of all article urls analyzed on the given <news site name>.
Each line contains a url and the then each of the feature values that we computed, all of which are tab separated.
The title line of this file contains the names of each feature value we recorded, in the form <feat number>_<gender number>_<feature name>.
Gender number is 0 for female and 1 for male.
