import math, collections

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.V = 0.0
    self.prev_word = "START" 
    self.prev_words = dict()
    self.bigrams = dict()
    self.unigrams = dict()
    self.probs = dict()
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """ 
    for sentence in corpus.corpus:  # iterate over sentences in the corpus
      for datum in sentence.data:   # iterate over datums in the sentence
        word = datum.word           # get the word

        word = str(word)
        # still need V
        if word in self.unigrams:
            self.unigrams[word] += 1.0
        else:
            self.unigrams[word] = 1.0
        
        # concat to get bigram
        bigram = self.prev_word + ' ' + word
        
        # build counts of prev word for the given bigram
        if self.prev_word in self.prev_words:
            self.prev_words[self.prev_word] += 1.0
        else:
            self.prev_words[self.prev_word] = 1.0
        
        # continue towards Count(bigram) + 1 
        if bigram in self.bigrams:
            self.bigrams[bigram] += 1.0
        else:
            self.bigrams[bigram] = 1.0
    
        self.V = len(self.unigrams)
        self.probs[bigram] = math.log(self.bigrams[bigram] / (self.prev_words[self.prev_word] + self.V))
        self.prev_word = word 

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    for token in sentence: # iterate over words in the sentence
        # retrieve Count(prev_word) of bigram
        if token in self.prev_words:
            c = self.prev_words[token]
        else:
            c = 1.0            
        bigram = self.prev_word + ' ' + token
        if bigram in self.bigrams:
            probability = self.bigrams[bigram]
        else:
            probability = math.log(1.0 / (c + self.V))
        
        score += probability 
        self.prev_word = token

    return score 
