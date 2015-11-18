#! /usr/bin/env python

## tested on python 2.7
# 8091 images, 6000 for training so far

import nltk
from nltk.util import ngrams
from nltk import FreqDist
#from nltk.corpus import stopwords
import re
import cPickle as pickle

NOUN = ('NN', 'NNS', 'NNP', 'NNPS')
VERB = ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
pos_tag = pickle.load(open('tnt_treebank_pos_tagger.pickle'))

class NGram(object):
    
    def __init__(self, imageIDs):
        self.imageIDs = imageIDs
        self.fdist1 = FreqDist()
        self.fdist2 = FreqDist()
        self.fdist3 = FreqDist()
        self.fdist_NV = FreqDist()
        self.fdist_VN = FreqDist()

    def loadCaptionText(self, textFile):
        """construct a dict, mapping image id to list of captions 
        This may be useful if we want to constrcut training data
        """
        with open(textFile) as f:
            for line in f:
                matchObj = re.match(r'(\w+_\w+)\.jpg#\d\s(.+)', line)
                if matchObj:
                    imageID = matchObj.group(1)
                    if imageID in self.imageIDs:
                        sentence = matchObj.group(2).lower()
                        sent_tokens = nltk.tokenize.word_tokenize(sentence)
                        if sent_tokens[-1] != '.':
                            sent_tokens.append('.')
                        #print sent_tokens
                        self.get_ngrams(sent_tokens)
                        #self.get_pairs(sent_tokens)
                else:
                    print "irregular entry: " + line
                    
        self.saveData()
    
    def saveData(self):
        with open('models/unigrams.pkl', 'w') as f:
            pickle.dump(self.fdist1, f)
        with open('models/bigrams.pkl', 'w') as f:
            pickle.dump(self.fdist2, f)            
        with open('models/trigrams.pkl', 'w') as f:
            pickle.dump(self.fdist3, f)            
        with open('models/noun_verb.pkl', 'w') as f:
            pickle.dump(self.fdist_NV, f)
        with open('models/verb_noun.pkl', 'w') as f:
            pickle.dump(self.fdist_VN, f)
            
    def get_ngrams(self, tokens):
        tokens.insert(0, '<START>')
        unigrams = ngrams(tokens,1)
        # key for unigrams is ('word',), not just 'word' string.
        for item in unigrams: self.fdist1[item] += 1 
        
        bigrams = ngrams(tokens,2)
        for item in bigrams: self.fdist2[item] += 1 
        
        trigrams = ngrams(tokens,3)
        for item in trigrams: self.fdist3[item] += 1 
        
    def get_pairs(self, tokens):
        tagged = pos_tag.tag(tokens)
        
        # get (noun, verb) pairs and (verb, noun) pairs
        for i in range(0, len(tagged)):
            if tagged[i][1] in NOUN:
                for j in range(i, len(tagged)):
                    if tagged[j][1] in VERB:
                        self.fdist_NV[(tagged[i][0], tagged[j][0])] += 1
            elif tagged[i][1] in VERB:
                for j in range(i, len(tagged)):
                    if tagged[j][1] in NOUN:
                        self.fdist_VN[(tagged[i][0], tagged[j][0])] += 1
        
def check_data(pkl_file):
    data = pickle.load(open(pkl_file))
    print data.viewitems()
    
    
if __name__== '__main__':
    srcFile = '../text/Flickr_8k.trainImages.txt'
    textFile = '../text/Flickr8k.lemma.token.txt'
    
    imageList = [] 
    with open(srcFile) as f:
        for line in f:
            if line:
                imageID = line.split('.')[0]
                imageList.append(imageID)
                ngram_generator = NGram(imageList)
                ngram_generator.loadCaptionText(textFile)