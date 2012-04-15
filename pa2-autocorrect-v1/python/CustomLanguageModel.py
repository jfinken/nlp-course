import math, collections

class CustomLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
        """  
        for sentence in corpus.corpus:
            previous_word = ""
            two_back_word = ""
            for datum in sentence.data:
                word = datum.word
                if word in self.unigrams:
                    self.unigrams[word] = self.unigrams[word] + 1.0
                else:
                    self.unigrams[word] = 1.0
                if previous_word != "":
                    bigram = previous_word + " | " + word
                    if bigram in self.bigrams:
                        self.bigrams[bigram] = self.bigrams[bigram] + 1.0
                    else:
                        self.bigrams[bigram] = 1.0
                        
                if two_back_word != "":
                    trigram = two_back_word + " | " + previous_word + " | " + word
                    if trigram in self.trigrams:
                        self.trigrams[trigram] = self.trigrams[trigram] + 1.0
                    else:
                        self.trigrams[trigram] = 1.0
                        
                two_back_word = previous_word
                previous_word = word

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 1.0
        unigram_vocabulary = len(self.unigrams) + 0.0
        previous_word = ""
        two_back_word = ""
        for word in sentence:
            unigram_lookup = self.unigrams[word] if word in self.unigrams else 0.0
            bigram = previous_word + " | " + word
            bigram_lookup = self.bigrams[bigram] if bigram in self.bigrams else 0.0
            trigram = two_back_word + " | " + previous_word + " | " + word
            trigram_lookup = self.trigrams[trigram] if trigram in self.trigrams else 0.0
            previous_word_lookup = self.unigrams[previous_word] if previous_word in self.unigrams else 0.0
            previous_bigram = two_back_word + " | " + previous_word
            previous_bigram_lookup = self.bigrams[previous_bigram] if previous_bigram in self.bigrams else 0.0
            
            # If we can't find a good trigram
            if trigram_lookup == 0.0 or previous_bigram_lookup == 0.0:
                # If we can't find a good bigram
                if bigram_lookup == 0.0 or previous_word_lookup == 0.0:
                    probability = 0.4 * (unigram_lookup + 1.0) / (unigram_lookup + unigram_vocabulary)
                else:
                    probability = (bigram_lookup / previous_word_lookup)
            else:
                probability = (trigram_lookup) / (previous_bigram_lookup)
            
            score += math.log(probability)
            
            two_back_word = previous_word
            previous_word = word
        return score
