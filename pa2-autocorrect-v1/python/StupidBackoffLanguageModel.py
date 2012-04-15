import math, collections

class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigrams = dict()
    self.bigrams = dict()
    self.total = 0

    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus: # iterate over sentences in the corpus
        prev = "";
        for datum in sentence.data: # iterate over datums in the sentence
            token = datum.word # get the word
            if token in self.unigrams:
                self.unigrams[token] += 1.0
            else:
                self.unigrams[token] = 1.0

            if prev != '':
                # build bigram
                bigram = prev + " | " + token 
                if bigram in self.bigrams:
                    self.bigrams[bigram] = self.bigrams[bigram] + 1.0
                else:
                    self.bigrams[bigram] = 1.0
            prev = token

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 1.0 
    unigram_vocabulary = len(self.unigrams) + 0.0
    prev = ""
    for word in sentence:
        unigram_lookup = self.unigrams[word] if word in self.unigrams else 0.0
        bigram = prev + " | " + word
        bigram_lookup = self.bigrams[bigram] if bigram in self.bigrams else 0.0
        previous_word_lookup = self.unigrams[prev] if prev in self.unigrams else 0.0

        # want a probability not a score
        
        # if no bigram, use an amount of the unigram
        if bigram_lookup == 0.0 or previous_word_lookup == 0.0:
            probability = 0.4 * (unigram_lookup + 1.0) / (unigram_lookup + unigram_vocabulary)
        else:
            probability = bigram_lookup / previous_word_lookup

        score += math.log(probability)
        prev = word
       
    return score 
