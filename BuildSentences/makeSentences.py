import pickle
import nltk
from nltk.util import ngrams
from nltk.probability import LidstoneProbDist
from random import shuffle
import math
import sys

from getNgrams import NOUN, VERB, pos_tag
from getSimilar import get_similar

class sentenceMaker(object):
    def __init__(self):
        #self.lambda1 = .2
        self.lambda2 = .6
        self.lambda3 = .4
        try:
            self._load_model()
        except:
            print "Cannot load model file from the current directory."
            sys.exit(1)
        self.memo = {}
#        try:
#            self.memo = pickle.load(open('models/memo.pkl'))
#        except:
#            self.memo = {}
    
    def _load_model(self):
        with open('unigrams.pkl') as f:
            self.pdist1 = LidstoneProbDist(pickle.load(f), 1)
        with open('bigrams.pkl') as f:
            self.pdist2 = LidstoneProbDist(pickle.load(f), 1)           
        with open('trigrams.pkl') as f:
            self.pdist3 = LidstoneProbDist(pickle.load(f), 0.05)          
        with open('noun_verb.pkl') as f:
            self.pdist_NV = LidstoneProbDist(pickle.load(f), 0.1)
        with open('verb_noun.pkl') as f:
            self.pdist_VN = LidstoneProbDist(pickle.load(f), 0.1)
    
    def find_synonyms(self, word):
        pass
    
    def expand_words(self, seeds, limit=5):
        """returns a list, every item is a list of tuples
            each tuple contains the seed
        """
        def penalty(x):
            if type(x) in (tuple, list) and len(x)>1:
                new_nouns = [x[0] for x in pos_tag.tag(x) if x[1] in NOUN+VERB and x[0] not in seeds]
                #print new_nouns
            return len(new_nouns)        
        
        fdist2 = self.pdist2.freqdist()
        fdist3 = self.pdist3.freqdist()
        output = []
        generated = []
        for seed in seeds:
            freqs = sorted([fdist2[x] for x in fdist2 if seed in x])
            if freqs:
                cutoff_freq = freqs[3*len(freqs)/4]
                #generated = [x for x in fdist2 if seed in x and fdist2[x]-0.5*penalty(x)*max_freq>=cutoff_freq][0:limit]
                generated = [x for x in fdist2 if (seed in x) and (not penalty(x)) and fdist2[x]>=cutoff_freq][0:limit]
            tmp = [seed]+generated

            freqs = sorted([fdist3[x] for x in fdist3 if seed in x])
            if freqs:
                cutoff_freq = freqs[3*len(freqs)/4]
                #generated = [x for x in fdist3 if seed in x and fdist3[x]-0.5*penalty(x)*max_freq>=cutoff_freq][0:limit*2]
                generated = [x for x in fdist3 if (seed in x) and (not penalty(x)) and fdist3[x]>=cutoff_freq][0:limit]
            tmp += generated
            output.append(tmp)
        return output
    
    @staticmethod
    def flatten(input_list):
        """ convert [(a,b),(c,d,e),f] to [a,b,c,d,e,f]
        """
        if not input_list:
            print "nothing to flatten"
            return []
        if type(input_list) in (str, unicode):
            return [input_list]
        output_list = []
        for item in input_list:
            if item == '': continue
            if type(item) in (str, unicode):
                output_list.append(item)
            elif len(item) > 1:
                output_list += [i for i in item]
            else:
                output_list.append(item)
        return output_list     
    
    def compute_score_rec(self, input_list):
        """recursively compute log prob of sentence (token list) """
        input_list = tuple(self.flatten(input_list))
        N = len(input_list)
        if N < 2:
            self.memo[input_list] = 0
            return 0
        
        if N == 2:
            score = self.pdist2.logprob(input_list) 
            return score
        
#        if len(input_list) == 3:
#            score = self.pdist3.logprob(input_list)
#            self.memo[input_list] = score
#            return score

        if input_list in self.memo:
            return self.memo[input_list]
        else:
            prev = self.compute_score_rec(input_list[0:-1]) #before adding new word/tuple
            delta3 = self.pdist3.logprob(input_list[-3:]) \
                        - self.pdist2.logprob(input_list[-3:-1])
            delta2 = self.pdist2.logprob(input_list[-2:]) \
                        - self.pdist1.logprob((input_list[-2],))
            score = prev + self.lambda3*delta3 + self.lambda2*delta2
            self.memo[input_list] = score
            return score
    
    def _save_memo(self):
        """save memo table to faciliate future computation """
        with open('models/memo.pkl','w') as f:
            pickle.dump(self.memo, f)
    
    def check_order(self, input_words):
        score = 0
        for i,first in enumerate(input_words[0:-1]):
            for second in input_words[i+1:]:
                score += self.pdist_NV.logprob((first, second))-self.pdist_NV.logprob((second, first))
                score += 2*( self.pdist_VN.logprob((first, second))-self.pdist_VN.logprob((second, first)) )
        return score
    
    def make_sentence_random(self, input_nouns, input_verbs=[], final=50, save_memo=False):
        self.input_nouns = input_nouns
        
        changed_nouns = [get_similar(x)[0][1] if get_similar(x) else x for x in input_nouns ]
        word_map = dict(zip(changed_nouns, input_nouns))
        changed_words = changed_nouns + input_verbs
        N = len(changed_words)
        print word_map
        
        expanded_ngrams = self.expand_words(changed_words, limit=20)
        print expanded_ngrams
        if N < 5:
            n_shuffles = 2*math.factorial(len(expanded_ngrams))
        else:
            n_shuffles = 50*N
        
        all_results = []
        word_order_score = None
        
        for i in range(len(changed_nouns)):
            expanded_ngrams.append(['','a','the'])
        expanded_ngrams.append(['','on','in','at','by','under','near', 'to'])
        for i in range(n_shuffles):
            shuffle(expanded_ngrams)
            expanded_ngrams_copy = expanded_ngrams[:] # must copy because python pass by reference
            this_score = self.check_order([x[0] for x in expanded_ngrams])
            if word_order_score:
                if this_score < word_order_score:
                    continue
            word_order_score = this_score
            result = self.make_sentence(expanded_ngrams_copy, keep=50*N, n_output=(N,4*N))
            all_results += result
        
        all_results = [(x[0], self.flatten(x[1]) ) for x in all_results] 
        #all_results = [(x[0]+sum(self.add_constraint(x[1], n_verbs=2, tokenized=True)), ' '.join(x[1]) ) for x in all_results]
        all_results = [(x[0]+self.check_order(x[1]), ' '.join(x[1]) ) for x in all_results]
        all_results = list(set(all_results))
        all_results.sort(reverse=True)
        
        words_resumed = []
        for item in all_results[0:final]:
            replaced = item[1]
            for key in word_map:
                replaced = replaced.replace(key, word_map[key])
                #new_nouns = [x[0] for x in pos_tag.tag(replaced.split()) if x[1] in NOUN and x[0] not in input_nouns]
                #penalty = 2*len(new_nouns)
            words_resumed.append( (item[0], replaced) )
        
        if save_memo:
            self._save_memo()
        return words_resumed
    
    def make_sentence(self, expanded_ngrams, keep=100, n_output=(5,8)):
        
        def scalor(x):
            if len(self.flatten(x)) <=n_output[0]:
                scalor = 1
            else:
                scalor = math.log(len(self.flatten(x)))/(1.2*math.log(n_output[0]))            
            return scalor
        

            
        
        expanded_ngrams.append(['.'])
#        for word in input_words:
#            if nltk.pos_tag([word])[0][1] in VERB:
#                seed_verbs.append(word)
        partial_sentences = []
        for choices in expanded_ngrams:
            #print 'choices:', choices
            if not partial_sentences:
                partial_sentences = [['<START>', x] for x in choices]
                continue
            candidates = []    
            for partial_sent in partial_sentences:
                for choice in choices:
                    tmp = partial_sent[:] # copy
                    tmp.append(choice)
                    candidates.append(tmp)
            candidates = [x for x in candidates if len(self.flatten(x))<=n_output[1]]

            scores = [self.compute_score_rec(x)/scalor(x) for x in candidates]
                
            pairs = sorted(zip(scores, candidates), reverse=True)
            if len(candidates) > keep:
                pairs = pairs[0:keep]
            partial_sentences = [x[1] for x in pairs]
        # return_list = [self.flatten(x) for x in partial_sentences[0:final] ]
        return pairs

    
    def tokens_log_prob(self, token_list, tokenized=True):
        if not tokenized:
            token_list = nltk.word_tokenize(token_list)
        # if token_list[0] != '<START>':
        #     token_list.insert(0, '<START>')
        unigrams = ngrams(token_list,1)
        bigrams = ngrams(token_list,2)
        trigrams = ngrams(token_list,3)
        
        #log_prob1 = sum( [self.pdist1.logprob(x) for x in unigrams] )bigrams.next()
        bigrams.next() # no need to use the first bigram
        log_prob2 = sum( [self.pdist2.logprob(x) for x in bigrams] )
        log_prob2 -= self.pdist2.logprob( (token_list[-2], token_list[-1]) ) # not need the last bigram
        #print 'bigrams log prob: ', log_prob2
        log_prob3 = sum( [self.pdist3.logprob(x) for x in trigrams] )
        #print 'trigrams log prob: ', log_prob3
        
        log_tokens = log_prob3 - log_prob2
        return log_tokens
    
    def select_best(self, sent_list, n=1, tokenized=False):
        scores = []
        for item in sent_list:
            scores.append(self.tokens_log_prob(item, tokenized) + sum(self.add_constraint(item, 1, tokenized)))
        d = dict(zip(scores, sent_list))
        scores.sort(reverse=True)
        return ([(d[score],score) for score in scores[0:n] ])

    def add_constraint(self, token_list, n_verbs=1, tokenized=False):
        if not tokenized:
            token_list = nltk.word_tokenize(token_list)
        tagged = nltk.pos_tag(token_list)
        
        # get (noun, verb) pairs and (verb, noun) pairs
        NV_pairs = []
        VN_pairs = []
        verbs_found = []
        for i in range(0, len(tagged)):
            if tagged[i][1] in NOUN:
                for j in range(i, len(tagged)):
                    if tagged[j][1] in VERB:
                        if (tagged[j][0] not in verbs_found) and (len(verbs_found) < n_verbs):
                            verbs_found.append(tagged[j][0])
                        if tagged[j][0] in verbs_found:
                            NV_pairs.append( (tagged[i][0], tagged[j][0]))
                        
            elif tagged[i][1] in VERB:
                for j in range(i, len(tagged)):
                    if tagged[j][1] in NOUN:
                        if (tagged[i][0] not in verbs_found) and (len(verbs_found) < n_verbs):
                            verbs_found.append(tagged[i][0])
                        if tagged[i][0] in verbs_found:
                            VN_pairs.append( (tagged[i][0], tagged[j][0]))                            
        if not verbs_found:
            return (-10000, -10000) # requires verbs  
                      
        log_NV = sum( [self.pdist_NV.logprob(x) for x in NV_pairs])
        log_VN = sum( [self.pdist_NV.logprob(x) for x in VN_pairs])
        return (log_NV, log_VN)

if __name__== '__main__':
    maker = sentenceMaker()
    nouns = ['girl', 'maillot','popsicle']
    verbs = ['chew']
    sents = maker.make_sentence_random(nouns, verbs, save_memo=False)
    for item in sents:
        print item      
