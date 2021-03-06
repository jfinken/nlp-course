#!/usr/bin/python

# NLP Programming Assignment #3
# NaiveBayes
# 2012

#
# The area for you to implement is marked with TODO!
# Generally, you should not need to touch things *not* marked TODO
#
# Remember that when you submit your code, it is not run from the command line 
# and your main() will *not* be run. To be safest, restrict your changes to
# addExample() and classify() and anything you further invoke from there.
#


import sys
import getopt
import os
import math
import collections
import operator

class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False 
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10

    self.totalDocs = 0.0
    self.V = 0.0          # total vocabulary size (unique tokens)
    self.Nc = dict()      # number of documents of klass
    self.textC = dict()   # concat of tokens (of all docs) of klass
    self.textCounts = dict()
    self.klasses = [] 
    self.condprob = dict()    # list of dict, thus making [][]
    self.unigrams = dict()

  #############################################################################
  # TODO TODO TODO TODO TODO 
  
  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    score = dict()
    score['pos'] = 0.0
    score['neg'] = 0.0

    # extract tokens of doc from V: 
    W = []
    for t in words:
      if t in self.unigrams:
        W.append(t)
   
    #print W 
    # apply
    for k in ['pos', 'neg']:
      #print '\tprior of %s: %s' % (k, str(self.Nc[k] / self.totalDocs) )
      score[k] = math.log( (self.Nc[k] / self.totalDocs) )
      for t in W:
        if t in self.condprob:
          if k in self.condprob[t]:
            score[k] += math.log(self.condprob[t][k])
            #print '\t\tcondprob of %s for %s: %s' % (t, k, str(self.condprob[t][k]) )

    print 'score[pos]: %s\tscore[neg]: %s' % (str(score['pos']), str(score['neg']))

    key_max = max(score.iteritems(), key=operator.itemgetter(1))[0]

    return key_max

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the NaiveBayes class.
     * Returns nothing

     Similar to: 
      TrainMultinomialNB(C, D) where C is a set of classes and D is a set of
      documents
    """
    if klass not in self.klasses:
      self.klasses.append(klass)

    # update text-c
    if klass not in self.textCounts:
      self.textCounts[klass] = dict()

    # update unigrams
    for token in words:
      if token in self.unigrams:
        self.unigrams[token] += 1.0
      else:
        self.unigrams[token] = 1.0
      
    for t in self.unigrams:
      if t not in self.textCounts[klass]:
        self.textCounts[klass][t] = 0.0
      self.textCounts[klass][t] += words.count(t)
    
    self.V = len(self.unigrams)

    # update textC 
    if klass in self.textC:
      self.textC[klass] += words      # concat words of the klass
    else:
      self.textC[klass] = words

    # update priors
    self.totalDocs += 1.0
    if klass in self.Nc:
        self.Nc[klass] += 1.0
    else:
        self.Nc[klass] = 1.0

    # now update the condition probabilities with add-one smoothing
    Tc = 0.0
    for k in self.klasses:
      for t in self.unigrams:
        if k in self.textC and k in self.textCounts:
          words = self.textC[k]
    #print 'unigram size: %d\tsize of text in klass[%s]: %d\tsize of V: %d' % (len(self.unigrams), klass, len(self.textC[klass]), self.V)

          # num occurrences of t in all text of klass  
          # - is probably the bottleneck: words can be upwards of 500,000 tokens!
          # - consider keeping a running count of the tokens in klass above...
          #Tc = words.count(t)
          Tc = 0.0
          if t in self.textCounts[k]:
            Tc = self.textCounts[k][t]

          if t in self.condprob:
            self.condprob[t][k] = (Tc + 1.0) / (len(words) + self.V)
          else:
            self.condprob[t] = dict()
            self.condprob[t][k] = (Tc + 1.0) / (len(words) + self.V)
    
    pass

  # TODO TODO TODO TODO TODO 
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)

  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits


  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      guess = self.classify(words)
      labels.append(guess)
    return labels
  
  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = [] 
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1: 
      print '[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir)

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName)) 
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName)) 
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered



def main():
  nb = NaiveBayes()
  (options, args) = getopt.getopt(sys.argv[1:], 'f')
  if ('-f','') in options:
    nb.FILTER_STOP_WORDS = True
  
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    accuracy = 0.0
    for example in split.train:
      words = example.words
      if nb.FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      classifier.addExample(example.klass, words)
  
    for example in split.test:
      words = example.words
      if nb.FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy

if __name__ == "__main__":
    main()
