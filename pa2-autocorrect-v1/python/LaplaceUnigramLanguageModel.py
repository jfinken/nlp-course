import math, collections

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.N = 0.0
    self.V = 0.0
    self.counts = dict()
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
        # computing Count(word) + 1:
        #   lookup the word, if exists, increment count for that word
        if word in self.counts:
            self.counts[word] += 1.0
        else:
            self.counts[word] = 1.0    

        self.N += 1.0
        self.V = len(self.counts)
        # compute running probabilities: Count(word) + 1 / (N + V)
        # self.probs[word] = math.log( self.counts[word] / (self.N + self.V) )

    #print "size of v: "+str(self.V)
    for word in self.counts:
      self.probs[word] = math.log( (self.counts[word]+1.0) / (self.N + self.V) )

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    for token in sentence: # iterate over words in the sentence
      # retrieve: (C(w) + 1) / (N + V), from model
      if token in self.probs:
        probability = self.probs[token]
      else:
        probability = math.log(1.0 / (self.N + self.V))
        
      score += probability

    return score 
