from nltk.corpus import wordnet as wn
import cPickle as pickle

def get_similar(word, n=1):
    """Given a word, get the most frequently used synonyms, hypernyms and hyponyms"""
    
    fdist = pickle.load(open('unigrams.pkl'))
    
    s = wn.synsets(word, pos=wn.NOUN)
    synos = []
    hyperset = []
    hyposet = []
    for item in s:
        synos += item.lemma_names()
        hyperset += item.hypernyms()
        hyposet += item.hyponyms()
    

    
    hypers = []
    for item in hyperset:
        hypers += item.lemma_names()

    hypos = []
    for item in hyposet:
        hypos += item.lemma_names()        
    
#    print 'synos: ', synos
#    print 'hypos: ', hypos
#    print 'hypers: ', hypers
        
    similar_set = set(synos+hypers+hypos)
    freq_word = [(fdist[(x,)],x) for x in similar_set]
    freq_word.sort(reverse=True)
    kept = min(len(freq_word), n)
        
    return freq_word[0:kept]

if __name__== '__main__':
    print get_similar('popsicle',n=4)