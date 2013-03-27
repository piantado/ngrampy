"""

	Note: This appears to be quicker than using "import gzip", even if we copy
	the gziped file to a SSD before reading. It's also faster than trying to change buffering on stdin
"""

import os
import re
import sys

import argparse
parser = argparse.ArgumentParser(description='MCMC for magic box!')
parser.add_argument('--out', dest='out-path', type=str, default="/tmp/", nargs="?", help='The file name for output (year appended)')
parser.add_argument('--year-bin', dest='year-bin', type=int, default=25, nargs="?", help='How to bin the years')
args = vars(parser.parse_args())

YEAR_BIN = int(args['year-bin'])
BUFSIZE = int(10e6) # We can allow huge buffers if we want...

prev_year,prev_ngram = None, None
count = 0

year2file = dict()
part_count = None

# python is not much slower than perl if we pre-compile regexes
cleanup = re.compile(r"(_[A-Za-z\_\-]+)|(\")") # kill tags and quotes
column_splitter = re.compile(r"[\t\s]") # split on tabs OR spaces, since some of google seems to use one or the other. 

for l in sys.stdin:
	
	l = l.lower().strip()
	l = cleanup.sub("", l)  
	#l = re.sub(r"\"", "", l) # remove quotes
	#l = re.sub(r"_[A-Z\_\-]+", "", l) # remove tags
	
	parts = column_splitter.split(l)
	if part_count is None: part_count = len(parts)
	if len(parts) != part_count: continue # skip this line if its garbage NOTE: this may mess up with some unicode chars?
	
	c = int(parts[-2])
	year = int(int(parts[-3]) / YEAR_BIN) * YEAR_BIN # round the year
	ngram = "\t".join(parts[0:-3]) # join everything else
	
	# output if different
	if year != prev_year or ngram != prev_ngram:
		
		if prev_year is not None:
			if prev_year not in year2file: year2file[prev_year] = open(args['out-path']+".%i"%prev_year, 'w', BUFSIZE)
			year2file[prev_year].write(  "%s\t%i\n" % (prev_ngram,count)  ) # write the year
		
		prev_ngram = ngram
		prev_year  = year
		count = c
	else:
		count += c
		
# And write the last line if we didn't alerady!
if year == prev_year and ngram == prev_ngram:
	year2file[prev_year].write("%s\t%i\n"%(ngram, c)) # write the year

# And close everything
for year in year2file.keys():
	year2file[year].close()
			
	
	
