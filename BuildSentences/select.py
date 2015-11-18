import makeSentences
import nltk

maker = makeSentences.sentenceMaker()
#with open('../text/testSents.txt') as f:
#    sent_list = f.readlines()

#result = maker.select_best(sent_list, n=8)
#for item in result:
#    print item

#print sent_list[1]
#expand = maker.expand_words(nltk.word_tokenize(sent_list[1]))
#print expand

#print maker.compute_score_rec(('a','boy','sitting'))
#print maker.compute_score_rec(('a','boy','sitting','on', 'the'))

#print maker.tokens_log_prob(['a','boy','sitting'])
#print maker.tokens_log_prob(['a','boy','sitting','on', 'the'], tokenized=True)

sents = maker.make_sentence_random(['girl', 'maillot','popsicle'],['chew'], save_memo=False)
#sents = maker.make_sentence_random(['dog', 'ball','grass'],['chase'], save_memo=False)
for item in sents:
    print item
