"""

	Note: This appears to be quicker than using "import gzip", even if we copy
	the gziped file to a SSD before reading. It's also faster than trying to change buffering on stdin
	
	pypy is MUCH faster
	
	TODO:
		- Clean this up to make play nicer with Tags. We can make it give NAs to nonexistent words!
		- Make this handle unicode correctly -- is it even wrong?
	Changes:
		- Sep 20 2013: this now outputs the tags as their own column, with NA for missing tags
"""

import os
import re
import sys
import itertools
import codecs
import gzip
import glob

import argparse
parser = argparse.ArgumentParser(description='Process google ngrams into year bins')
parser.add_argument('--in', dest='in', type=str, default=None, nargs="?", help='The file name for input')
parser.add_argument('--out', dest='out', type=str, default="/tmp/", nargs="?", help='The file name for output (year appended)')
parser.add_argument('--year-bin', dest='year-bin', type=int, default=10, nargs="?", help='How to bin the years')
parser.add_argument('--quiet', dest='quiet', default=False, action="store_true", help='Output tossed lines?')
args = vars(parser.parse_args())

YEAR_BIN = int(args['year-bin'])
BUFSIZE = int(1e6) # We can allow huge buffers if we want...
ENCODING = 'utf-8'

prev_year,prev_ngram = None, None
count = 0

year2file = dict()
part_count = None

# python is not much slower than perl if we pre-compile regexes

#cleanup = re.compile(r"(_[A-Za-z\_\-]+)|(\")") # The old way -- delete tags and quotes
line_splitter = re.compile(r"\n", re.U)
cleanup_quotes = re.compile(r"(\")", re.U) # kill quotes
#column_splitter = re.compile(r"[\s]", re.U) # split on tabs OR spaces, since some of google seems to use one or the other. 

tag_match = re.compile(r"^(.+?)(_[A-Z\_\-\.\,\;\:]+)?$", re.U) # match a tag at the end of words (assumes 
def tagify(x):
	"""
		Take a word with a tag ("man_NOUN") and give back ("man","NOUN") with "NA" if the tag is not there
	"""
	m = re.match(tag_match, x)
	if m:
		g = m.groups()
		if g[1] is None: return (g[0], "NA")
		else:            return g
	else: return []

def chain(args):
	a = []
	for x in args: a.extend(x)
	return a
	

for f in glob.glob(args['in']):
	
	# Unzip and encode
	inputfile = gzip.open(f, 'r')
	for l in inputfile:
		#l = l.decode(ENCODING)
		
		l = l.strip() ## To collapse case
		l = cleanup_quotes.sub("", l)   # remove quotes
		#print >>sys.stderr, l
		
		#parts = column_splitter.split(l)
		parts = l.split() # defaultly should handle splitting on whitespace, much friendlier with unicode
		
		# Our check on the number of parts -- we require everything to have as many as the frist line
		if part_count is None: part_count = len(parts)
		if len(parts) != part_count: 
			if not args['quiet']: print "Wrong number of items on line: skiping ", l, parts
			continue # skip this line if its garbage NOTE: this may mess up with some unicode chars?
			
		c = int(parts[-2]) # the count
		year = int(int(parts[-3]) / YEAR_BIN) * YEAR_BIN # round the year
		ngram_ar = chain(map(tagify,parts[0:-3]))
		if all([x != "NA" for x in ngram_ar]): # Chuck lines that don't have all things tagged
			ngram = "\t".join(chain(map(tagify,parts[0:-3]))) # join everything else, including the tags separated out
		else: continue
		
		# output if different
		if year != prev_year or ngram != prev_ngram:
			
			if prev_year is not None:
				if prev_year_s not in year2file: 
					year2file[prev_year_s] = open(args['out']+".%i"%prev_year, 'w', BUFSIZE)
				year2file[prev_year_s].write( "%s\t%i\n" % (prev_ngram,count)  ) # write to the year file TODO: This might should be unicode fanciness?
			
			prev_ngram = ngram
			prev_year  = year
			prev_year_s = str(prev_year)
			count = c
		else:
			count += c
		
		# And write the last line if we didn't alerady!
		if year == prev_year and ngram == prev_ngram:
			if prev_year_s not in year2file: 
				year2file[prev_year_s] = open(args['out']+".%i"%prev_year, 'w', BUFSIZE)
			year2file[prev_year_s].write( "%s\t%i\n" % (prev_ngram,count)  ) # write to the year file TODO: This might should be unicode fanciness?
			
	inputfile.close()
	
# And close everything
for year in year2file.keys():
	year2file[year].close()
			
	
	
