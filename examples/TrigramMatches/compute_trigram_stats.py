
from ngrampy.LineFile import *
import os
GOOGLE_ENGLISH_DIR = "/home/piantado/Desktop/mit/Corpora/GoogleNGrams/3/"
VOCAB_FILE = "Vocabulary/EnglishVocabulary.txt"

# Read the vocabulary file
vocabulary = [ l.strip() for l in open(VOCAB_FILE, "r") ]

#rawG = LineFile(["test3.txt"], header=["w1", "w2", "w3", "cnt123"]) # for debugging
rawG = LineFile([GOOGLE_ENGLISH_DIR+x for x in os.listdir(GOOGLE_ENGLISH_DIR)], header=["w1", "w2", "w3", "cnt123"]) 

rawG.clean() # already done!
rawG.restrict_vocabulary("w1 w2 w3", vocabulary) # in fields w1 and w2, restrict our vocabulary
rawG.sort(keys="w1 w2 w3") # Since we collapsed case, etc. This could also be rawG.sort(keys=["w1","w2","w3"]) in the other format.
rawG.resum_equal("w1 w2 w3", "cnt123" )

# Where we store all lines
G = rawG.copy()

# Now go through and compute what we want
G1 = rawG.copy() # start with a copy
G1.delete_columns( "w2 w3" ) # delete the columns we don't want
G1.sort("w1" ) # sort this by the one we do want 
G1.resum_equal( "w1", "cnt123" ) # resum equal
G1.rename_column("cnt123", "cnt1") # rename the column since its now a sum of 1
G.sort("w1") # sort our target by w
G.merge(G1, keys1="w1", tocopy="cnt1") # merge in
G1.delete() # and delete this temporary

G2 = rawG.copy()
G2.delete_columns( "w1 w3" )
G2.sort("w2" )
G2.resum_equal( "w2", "cnt123" )
G2.rename_column("cnt123", "cnt2")
G.sort("w2")
G.merge(G2, keys1="w2", tocopy="cnt2")
G2.delete()

G3 = rawG.copy()
G3.delete_columns( "w1 w2" )
G3.sort("w3")
G3.resum_equal( "w3", "cnt123" )
G3.rename_column("cnt123", "cnt3")
G.sort("w3")
G.merge(G3, keys1="w3", tocopy="cnt3")
G3.delete()

G12 = rawG.copy()
G12.delete_columns( ["w3"] )
G12.sort("w1 w2" )
G12.resum_equal( "w1 w2", "cnt123" )
G12.rename_column("cnt123", "cnt12")
G.sort("w1 w2") # do this for merging
G.merge(G12, keys1="w1 w2", tocopy=["cnt12"])
G12.delete()

G23 = rawG.copy()
G23.delete_columns( ["w1"] )
G23.sort("w2 w3" )
G23.resum_equal( "w2 w3", "cnt123" )
G23.rename_column("cnt123", "cnt23")
G.sort("w2 w3") # do this for merging
G.merge(G23, keys1="w2 w3", tocopy=["cnt23"])
G23.delete()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Now compute all the arithmetic, etc. 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Make a colum: call it unigram, a function of three arguments, and give it w1,w2,w3 as arguments
from math import log
def log2(x): log(x,2.0)

def logsum(*x): return str(round(sum(map(log,map(float,x))), 4)) # must take a string and return a string
#def logcol(x) : return logsum([x])
G.make_column("unigram", logsum, "cnt1 cnt2 cnt3")
G.make_column("bigram",  logsum, "cnt12 cnt23")
G.make_column("trigram",  logsum, "cnt123")

G.sort("unigram bigram trigram", dtype=float)

##G.cat()
G.head()
