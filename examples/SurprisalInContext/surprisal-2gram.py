"""
	This file shows how to use ngrampy to compute the average surprisal measures from Piantadosi, Tily & Gibson (2011)
	
	python surprisal-2gram.py --in=/path/to/google --path=/tmp/temporaryfile > surprisal.txt
	
"""
from ngrampy.LineFile import *
import os
import argparse
import glob

ASSERT_SORTED = True # if you want an extra check on sorting

parser = argparse.ArgumentParser(description='Compute average surprisal from google style data')
parser.add_argument('--in', dest='in', type=str, default="/home/piantado/Desktop/mit/Corpora/GoogleNGrams/2/*", nargs="?", help='The directory with google files (e.g. Google/3gms/)')
parser.add_argument('--path', dest='path', type=str, default="/tmp/GoogleSurprisal", nargs="?", help='Where the database file lives')
args = vars(parser.parse_args())	

print "# Loading files"
G = LineFile( glob.glob(args['in']), header=["w1", "w2", "cnt12"], path=args['path']) 
print "# Cleaning"
G.clean(columns=3)

# Since we collapsed case, go through and re-sum the triple counts
print "# Resumming for case collapsing"
G.sort(keys="w1 w2") 
G.resum_equal("w1 w2", "cnt12", assert_sorted=ASSERT_SORTED ) # in collapsing case, etc., we need to re-sum

# Now go through and
print "# Making marginal counts"
G.make_marginal_column("cnt1", "w1", "cnt12") 

# and compute surprisal
print "# Sorting by word"
G.sort("w2")

print "# Computing surprisal"
G.print_average_surprisal("w2", "cnt12", "cnt1", assert_sorted=ASSERT_SORTED)

# And remove my temporary file:
print "# Removing my temporary file"
G.delete_tmp()

# If you have a file that's already sorted, etc:
#G = LineFile(["/ssd/GoogleSurprisal-ALREADYFILTERED"], path="/ssd/Gsurprisal", header=["w1", "w2", "w3", "cnt12", "cnt12"])  # for debugging
#G.print_average_surprisal("w3", "cnt12", "cnt12", assert_sorted=False)
