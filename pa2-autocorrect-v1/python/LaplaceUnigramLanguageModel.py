class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.words = set([][])
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus: # iterate over sentences in the corpus
      for datum in sentence.data: # iterate over datums in the sentence
        word = datum.word # get the word

        # computing C(word) + 1:

        # lookup the word, if exists, increment count for that word
        # else
          #self.words.add(word)
          #increment

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    for token in sentence: # iterate over words in the sentence
      # compute: (C(w) + 1) / (N + V), retrieving C(w) from trained matrix
      #probability = math.log(1.0/len(self.words))
      #score += probability

    return 0.0
