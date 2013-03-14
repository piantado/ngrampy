"""
	This file is for constructing stimuli which are matched on bigram and unigram surprisal
	It requires you to have run compute_trigram_stats and output the result to /ssd/trigram-stats
	It also takes a bad word file to filter out bad words from our experimental stimuli
"""

from ngrampy.LineFile import *
import os
SUBSAMPLE_N = 15000
tolerance = 0.001
BAD_WORD_FILE = "badwords.txt"

def check_tolerance(x,y):
	"""
		A handy function to check if some variables are within tolerance percent of each other
	"""
	return abs(x-y) / ((x+y)/2.) < tolerance

# This will copy the file, make a new one, and then print out possible lines
G = LineFile(files=["/ssd/trigram-stats"], path="/ssd/subsampled-stimuli", header="w1 w2 w3 c123 c1 c2 c3 c12 c23 unigram bigram trigram")

# Now throw out the porno words
porno_vocabulary = [ l.strip() for l in open(BAD_WORD_FILE, "r") ]
G.restrict_vocabulary("w1 w2 w3", porno_vocabulary, invert=True)

# and then subsample
G.subsample(N=SUBSAMPLE_N)

# and make sure we are sorted for the below
G.sort("unigram bigram trigram", dtype=float)
G.head() # just a peek

item_number = 0
line_stack = []
for l in G.lines(tmp=False, parts=False):
	# extrac the columns from line
	unigram, bigram, trigram =  G.extract_columns(l, keys="unigram bigram trigram", dtype=float)
	
	# now remove things which cannot possibly match anymore
	while len(line_stack) > 0 and not check_tolerance(unigram, G.extract_columns(line_stack[0], keys="unigram", dtype=float)[0]):
		del line_stack[0]
	
	# now go through the line_stack and try out each 
	# it must already be within tolerance on unigram, or it would have been removed
	for x in line_stack:
		#print "Checking ", x
		x_unigram, x_bigram, x_trigram =  G.extract_columns(x, keys="unigram bigram trigram", dtype=float)
		
		# it must have already been within tolerance on unigram or it would be removed
		assert( check_tolerance(unigram, x_unigram) ) 
		
		# and check the bigrams
		if check_tolerance(bigram, x_bigram):
			print len(line_stack), item_number, l
			print len(line_stack), item_number, x
			item_number += 1
		
	# and add this on
	line_stack.append(l)
