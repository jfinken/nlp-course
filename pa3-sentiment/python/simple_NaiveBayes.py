#!/usr/bin/python

import sys
import getopt
import os
import math
import collections
import operator

klasses=['china','not-china']
totalDocs = 0.0
V = 0.0          # total vocabulary size (unique tokens)
Nc = dict()      # number of documents of klass
textC = dict()   # concat of tokens (of all docs) of klass
textCounts = dict()
condprob = dict()    # list of dict, thus making [][]
unigrams = dict()
score = dict()
score['china'] = 0.0
score['not-china'] = 0.0

docs = [
    {'china': ['chinese', 'beijing', 'chinese']},
    {'china': ['chinese', 'chinese', 'shanghai']},
    {'china': ['chinese', 'macao']},
    {'not-china': ['tokyo', 'japan', 'chinese']}
]

print 'training...'
for doc in docs:    
  klass = doc.keys()[0]
  print '\nklass: '+klass
  # update text-c
  if klass not in textCounts:
    textCounts[klass] = dict()

  # for each token of the document
  words = doc.values()[0]
  for token in words:
    if token in unigrams:
      unigrams[token] += 1.0
    else:
      unigrams[token] = 1.0
 

  for t in unigrams:
    if t not in textCounts[klass]:
      textCounts[klass][t] = 0.0
    textCounts[klass][t] += words.count(t)
    
  V = len(unigrams)

  # update textC 
  if klass in textC:
    textC[klass] += words      # concat words of the klass
  else:
    textC[klass] = words

  # update priors
  totalDocs += 1.0
  if klass in Nc:
    Nc[klass] += 1.0
  else:
    Nc[klass] = 1.0
  #print 'total-docs: %d' % totalDocs
  #print Nc

  # now update the condition probabilities with add-one smoothing
  Tc = 0.0
  #print 'unigram size: %d\tsize of text in klass[%s]: %d\tsize of V: %d' % (len(unigrams), klass, len(textC[klass]), V)

  for k in klasses:
    for t in unigrams:
      if k in textC and k in textCounts:
        words = textC[k]
        # num occurrences of t in all text of klass  
        Tc = 0.0
        if t in textCounts[k]:
          Tc = textCounts[k][t]

        ########################################################
        # bug: V is updated if/when the klass changes
        #      but it is never updated in condprob[t][klass]
        #      be careful with foreach klass
        ########################################################
        if t == 'chinese':
        #print textCounts
          print 'numer: %d'% (Tc + 1.0)
          print 'len(words): %d\tV: %d'% (len(words), V)

        if t in condprob:
          condprob[t][k] = (Tc + 1.0) / (len(words) + V)
        else:
          condprob[t] = dict()
          condprob[t][k] = (Tc + 1.0) / (len(words) + V)

  
print condprob

################################################################
# NB application
################################################################
print 'applying NB model...'
words = ['chinese', 'chinese', 'chinese', 'tokyo', 'japan']
# extract tokens of doc from V: 
W = []
for t in words:
  if t in unigrams:
    W.append(t)
   
# apply
for k in klasses:
  #print '\tprior of %s: %s' % (k, str(Nc[k] / totalDocs) )
  score[k] = (Nc[k] / totalDocs) 
  # using W is buggy
  for t in words:
    if t in condprob:
      if k in condprob[t]:
        score[k] *= condprob[t][k]
        #print '\t\tcondprob of %s for %s: %s' % (t, k, str(condprob[t][k]) )

print 'score[china]: %s\tscore[not-china]: %s' % (str(score['china']), str(score['not-china']))

#key_max = max(score.iteritems(), key=operator.itemgetter(1))[0]
