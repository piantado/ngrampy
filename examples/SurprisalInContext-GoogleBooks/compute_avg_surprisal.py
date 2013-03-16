"""
	This file shows how to use ngrampy to compute the average surprisal measures from Piantadosi, Tily & Gibson (2011)
	
	python compute_average_surprisal --in=/path/to/google --path=/tmp/temporaryfile > surprisal.txt
	
"""
from ngrampy.LineFile import *
import os
import argparse

#import psyco 
#psyco.jit() 
#from psyco.classes import *

ASSERT_SORTED = False # if you want an extra check on sorting

parser = argparse.ArgumentParser(description='Compute average surprisal from google style data')
parser.add_argument('--in', dest='in', type=str, default="/home/piantado/Desktop/mit/Corpora/GoogleNGrams/3/", nargs="?", help='The directory with google files (e.g. Google/3gms/')
parser.add_argument('--path', dest='path', type=str, default="/tmp/GoogleSurprisal", nargs="?", help='Where the database file lives')
args = vars(parser.parse_args())	

print "# Loading files"
G = LineFile([args['in']+"/"+x for x in os.listdir(args['in'])], header=["w1", "w2", "w3", "year", "cnt123", "volcnt"], path=args['path']) 
print "# Cleaning"
G.clean()

# Since we collapsed case, go through and re-sum the triple counts
print "# Resumming out year, case collapsing"
G.sort(keys="w1 w2 w3") 
G.resum_equal("w1 w2 w3", "cnt123", assert_sorted=ASSERT_SORTED ) # in collapsing case, etc., we need to re-sum

# Now go through and 
Gcontext = G.copy()
Gcontext.delete_columns( "w3" ) # delete the columns we don't want
print "# Sorting by context"
Gcontext.sort("w1 w2") # sort this by the one we do want 
print "# Computing context sum"
Gcontext.resum_equal( "w1 w2", "cnt123", assert_sorted=ASSERT_SORTED ) # resum equal
Gcontext.rename_column("cnt123", "cnt12") # rename the column since its now a sum of 1
print "# Sorting by context"
Gcontext.sort("w1 w2") # sort our target by w
print "# Merging"
G.merge(Gcontext, keys1="w1 w2", tocopy="cnt12", assert_sorted=ASSERT_SORTED) # merge in
#Gcontext.delete() # and delete this temporary

# and compute surprisal
print "# Sorting by word"
G.sort("w3")

print "# Computing surprisal"
G.print_average_surprisal("w3", "cnt123", "cnt12", assert_sorted=ASSERT_SORTED)

# If you have a file that's already sorted, etc:
#G = LineFile(["/ssd/GoogleSurprisal-ALREADYFILTERED"], path="/ssd/Gsurprisal", header=["w1", "w2", "w3", "cnt123", "cnt12"])  # for debugging
#G.print_average_surprisal("w3", "cnt123", "cnt12", assert_sorted=False)