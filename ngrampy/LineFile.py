
""" 
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	This class allows manipulation of google ngram data (and similar formatted) data. 
	When you call functions on LineFiles, the changes are echoed in the file. 
	
	The uses tab (\t) as the column separator. 
	
	When you run this, if you get an encoding error, you may need to set the environment to 
	
		export PYTHONIOENCODING=utf-8
		
		
	
	TODO: 
		- Make this so each function call etc. will output what it did
	NOTE:
		- Column names cannot contain spaces. 
		- do NOT change the printing to export funny, because then it will collapse characters to ascii in a bad way
		
	Steve Piantadosi 2012
	Licensed under GPL 3.0
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.

	
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
import sys
import re
import unicodedata
import heapq
import shutil
import random
import codecs # for writing utf-8

from math import log
from collections import defaultdict
from copy import deepcopy

# A temporary file like /tmp
NGRAMPY_DEFAULT_PATH = "/tmp" #If no path is specified, we go here

ECHO_SYSTEM = True # show the system calls we make?
CLEAN_TMP = False # when we create temporary files, do we remove them when we're done? (sorting files are always removed)
SORT_DEFAULT_LINES = 10000000 # how many lines to sorted at a time in RAM when we sort a large file?
ENCODING = 'utf-8'

spaceRegex = re.compile("\s") # for splitting on spaces, etc.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Some helpful functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def read_and_parse(inn, keys):
		"""
			Read a line and parse it by tabs, returning the line, the tab parts, and some columns
		"""
		line = inn.readline().strip()
		if not line: return line, None, None
		else:
			parts = re.split("\t", line)
			return line, parts, "\t".join([parts[x] for x in keys])

def systemcall(x):
	"""
		Call System functions but echo if we want
	"""
	if ECHO_SYSTEM: print >>sys.stderr, x
	os.system(x)
		
def ifelse(x,y,z):
	if x: return y
	else: return z

def listifnot(x):
	if isinstance(x, list): return x
	else:                   return [x]
	
def myassert(tf, s):
	if not tf: print "*** Assertion fail: ",s
	assert(tf)
	
def log2(x): return log(x,2.)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Class definition
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class LineFile:
	
	def __init__(self, files, header=None, path=None, force_nocopy=False):
		"""
			Createa  new file object with the specified header. It takes a list of files
			and cats them to tmppath (overwriting it). If testing, it only does the first
			one 
			
			header - give each column a name (you can refer by name instead of number)
			path   - where is this file stored? If None, we make a new temporary files
			force_nocopy - Primarily for debugging, this prevents us from copying a file and just uses the path as is
					you should pass files a list of length 1 which is the file, and path should be None
					as in, LineFile(files=["/ssd/trigram-stats"], header="w1 w2 w3 c123 c1 c2 c3 c12 c23 unigram bigram trigram", force_nocopy=True)
		"""
		if force_nocopy:
			assert(len(files) == 1)
			self.path = files[0]
			
		else:
			if path is None:
				self.path = NGRAMPY_DEFAULT_PATH+"/tmp" # used by get_new_path, since self.path is where we get the dir from
				self.path = self.get_new_path() # overwrite iwth a new file
			else:
				self.path = path
				
			# if it exists, let's just move it to a backup name
			if os.path.exists(self.path):
				systemcall("mv "+self.path+" "+self.path+".old")
				
			# and if we specified a bunch of input files
			for f in files:
				if re.search(".idx$",f): continue # skip the index files
				if re.search(".gz$", f):    systemcall("gunzip -d -c "+f+" >> "+self.path)
				elif re.search(".bz2$", f): systemcall("bzip2 -d -c "+f+" >> "+self.path)
				else:                       systemcall("cat "+f+" >> "+self.path)
		
		# just keep track
		self.files = files
		
		# and store some variables
		self.tmppath = self.path+".tmp"
		
		if isinstance(header, str): header = spaceRegex.split(header)
		self.header = header
		
	def setheader(self, *x): self.header = x
	def rename_column(self, x, v): self.header[self.to_column_number(x)] = v
		
	def to_column_number(self, x):
		"""
		 Takes either:
			a column number - just echoes back
			a string - returns the right column number for the string
			a whitespace separated string - returns an array of column numbers
			an array - maps along and returns
		 
		"""
		
		if isinstance(x, int):    return x
		elif isinstance(x, list): return map(self.to_column_number, x)
		elif isinstance(x, str): 
		
			if spaceRegex.search(x):  # if spaces, treat it as an array and map
				return map(self.to_column_number, spaceRegex.split(x))
			
			# otherwise, a single string so just find the header that equals it
			for i in xrange(len(self.header)):
				if self.header[i] == x: return i
		
		print >>sys.stderr, "Invalid header name ["+x+"]", self.header
		exit(1)
			
		
	def delete_columns(self, cols):
		
		# make sure these are increasing so we can delete in order
		cols = sorted(listifnot(self.to_column_number(cols)), reverse=True)
		
		self.mv_tmp()
		o = codecs.open(self.path, "w", ENCODING)
		for parts in self.lines(parts=True):
			for c in cols: del parts[c]
			print >>o, "\t".join(parts)
		o.close()
		
		# and delete from the header
		if self.header is not None:
			for c in cols: del self.header[c]
		
		
	def copy(self, path=None):
		
		if path is None: path = self.get_new_path() # make a new path if its not specified
		
		# we can just copy the file by treating it as one of the "files"
		# and then use this new path, not the old one!
		return LineFile([self.path], header=deepcopy(self.header), path=path)
	
	def get_new_path(self): 
		ind = 1
		while True:
			path = os.path.dirname(self.path)+"/ngrampy-"+str(ind)
			if not os.path.isfile(path): return path
			ind += 1
			
	def mv_tmp(self):
		"""
			Move myself to my temporary file, so that I can cat to my self.path
		"""
		shutil.move(self.path, self.tmppath)
		
	def rename(self, n):
		shutil.move(self.path, n)
		self.path = n
		
	def rm_tmp(self):
		"""
			Remove the temporary file
		"""
		os.remove(self.tmppath)
	def cp(self, f): shutil.cp(self.path, f)
	
	def extract_columns(self, line, keys, dtype=unicode, sep="\t"):
		"""'ut
			Extract some columns from a single line. Assumes that keys are numbers (e.g. already mapped through to_column_number)
			and wil return the columns as the specified dtype
			NOTE: This always returns a list, even if one column is specified. This may change in the future
			
			e.g. line="a\tb\tc\td"
			     keys=[1,4]
			     gives: ["a", "b", "c", "d"], "b\td"
		"""
		if isinstance(keys, str): keys = listifnot(self.to_column_number(keys))
		
		parts = re.split(sep, line)
		
		if isinstance(dtype,list):
			return [ dtype[i](parts[x]) for i,x in enumerate(keys)]
		else: 
			return [ dtype(parts[x]) for x in keys ]
			

		
	def clean(self, tab=True, lower=True, alphanumeric=True, count_columns=True):
		"""
			This does several things:
				tab - Replace all whitspace on each line with tabs
				lower - convert to lowercase
				alphanumeric - toss lines with non-letter category characters (in unicode)
				count_columns - if True, we throw out rows that don't have the same number of columns as the first line
			
		"""
		self.mv_tmp()
		
		col_count = None
		
		o = codecs.open(self.path, "w", ENCODING)
		collapser = re.compile("[\d\s]", re.UNICODE) # for filtering out numbers and whitespace
		for l in self.lines():
			keep = True
			if alphanumeric: # if we require alphanumeric
				collapsed = collapser.sub("", l) # replace digits and spaces with nothing so allw e have are characters
				for k in collapsed:		
					n = unicodedata.category(k)
					if n == "Ll" or n == "Lu": pass
					else: 
						keep = False # throw out lines with non-letter categories
						break
			
			if not keep: continue # we can skip early here if we want
			# clean up according to specs
			if tab: l = re.sub("\s", "\t", l)
			if lower: l = l.lower()
			
			# check the number of columns
			if count_columns: 
				cn = len(re.split("\t", l))
				if col_count is None: 
					col_count = cn # save the first line
				elif col_count != cn:
					keep = False
					print >>sys.stderr, "# Tossed line with bad column count (",cn,"):", l
			
			# and print if we should keep it
			if keep: print >>o, l
			
		o.close()
		
		if CLEAN_TMP: self.rm_tmp()
		
	def restrict_vocabulary(self, cols, vocabulary, invert=False):
		"""
			Make a new version where "cols" contain only words matching the vocabulary
			OR if invert=True, throw out anything matching cols
		"""
		
		cols = listifnot(self.to_column_number(cols))
		
		vocabulary_d = dict()
		for v in vocabulary: vocabulary_d[v] = True
		
		self.mv_tmp()
		o = codecs.open(self.path, "w", ENCODING)
		for l in  self.lines():
			parts = re.split('\t', l)
			keep = True
			for c in cols: # for each thing to check, check it!
				if vocabulary_d.get(parts[c], False) is invert:  # keep things in vocabulary unless invert
					keep = False
					break
			if keep: print >>o, l
		o.close()
		
		if CLEAN_TMP: self.rm_tmp()
		
	
	def resum_equal(self, keys, sumkeys, assert_sorted=True):
		"""
			Takes all rows which are equal on the keys and sums the remaining keys. 
			Anything not in keys or sumkeys, there are no guarantees for
		"""
		self.mv_tmp()
		
		keys    = listifnot(self.to_column_number(keys))
		sumkeys = listifnot(self.to_column_number(sumkeys))
		
		if assert_sorted:
			self.assert_sorted(keys,  allow_equal=True)
		
		o = codecs.open(self.path, "w", ENCODING)
		old_compkey = None
		old_parts = None
		sums = defaultdict(int)
		for parts in self.lines(parts=True):

			# what key do we use to compare equality?
			compkey = "\t".join([parts[x] for x in keys])
			
			if compkey == old_compkey: # sum up:
			
				for x in sumkeys: sums[x] += int(parts[x])
				
			else: # else print out the previous line, if nothing
				
				if old_parts is not None:
					# the easiest way to do this is just to overwrite in parts with the new counts
					for x in sumkeys: old_parts[x] = str(sums[x])
					
					# and print output					
					print >>o, "\t".join(old_parts)
				
				# and update the new 
				sums = defaultdict(int)
				for x in sumkeys: sums[x] += int(parts[x])
				
				#print (old_compkey<compkey), old_compkey, "=?=", compkey
				assert(old_compkey < compkey) # better be in sorted order, or this fails
				old_compkey = compkey
				
				old_parts = parts
		
		# and print the last line:
		for x in sumkeys: old_parts[x] = str(sums[x])
		print >>o, "\t".join(old_parts)
		
		o.close()
	
		if CLEAN_TMP: self.rm_tmp()
		
	def assert_sorted(self, keys, dtype=unicode, allow_equal=False):
		"""
			Assert that a file is sorted by certain columns
			This good for merging, etc., which optionally check requirements 
			to be sorted
		"""
		keys = self.to_column_number(keys)
	
		prev_sortkey = None
		for line in codecs.open(self.path, "r", ENCODING):
			line = line.strip()
			sortkey = self.extract_columns(line, keys=keys, dtype=dtype) # extract_columns gives them back tab-sep, but we need to cast them
			
			if prev_sortkey is not None:
				if allow_equal: myassert( prev_sortkey <= sortkey, line+";"+unicode(prev_sortkey)+";"+unicode(sortkey) )
				else:           myassert( prev_sortkey < sortkey, line+";"+unicode(prev_sortkey)+";"+unicode(sortkey) )
			
			prev_sortkey = sortkey
	

	def cat(self): systemcall("cat "+self.path)
	def head(self, n=10): 
		print self.header
		systemcall("head -n "+unicode(n)+" "+self.path)
	def delete(self):
		os.remove(self.path)
		os.remove(self.tmppath)
		
	def make_column(self, newname, function, keys):
		"""
			Make a new column as some function of the other rows
			make_column("unigram", lambda x,y: int(x)+int(y), "cnt1 cnt2")
			will make a column called "unigram" that is the sum of cnt1 cnt2
			
			NOTE: The function MUST take strings and return strings, or else we die
			
			newname - the name for the new column. You can pass multiple if function returns tab-sep strings
			function - a function of other row arguments. Must return strings
			args - column names to get the arguments
		"""
		
		self.mv_tmp()
		
		keys = listifnot( self.to_column_number(keys) )
		
		o = codecs.open(self.path, "w", ENCODING)

		for line in self.lines():
			parts = re.split("\t", line)
			print >>o, line+"\t"+function(*[parts[i] for i in keys])
			
		self.header.extend(listifnot(newname))
		o.close()
		
		if CLEAN_TMP: self.rm_tmp()
		
	def sort(self, keys, lines=SORT_DEFAULT_LINES, dtype=unicode, reverse=False):
		"""
			Sort me by my keys. this breaks the file up into subfiles of "lines", sorts them in RAM, 
			and the mergesorts them
			
			We could use unix "sort" but that gives weirdness sometimes, and doesn't handle our keys
			as nicely, since it treats spaces in a counterintuitive way
			
			dtype - the type of the data to be sorted. Should be a castable python type
			        e.g. str, int, float
		"""
		self.mv_tmp()
		
		temp_id = 0
		current_lines = []
		sorted_tmp_files = [] # a list of generators, yielding each line of the file
		
		keys = listifnot(self.to_column_number(keys))
		
		# a generator to hand back lines of a file and keys for sorting
		def yield_lines(f):
			for l in codecs.open(f, "r", ENCODING): yield get_sort_key(l.strip())
				
		# Map a line to sort keys (e.g. respecting dtype, etc) ; we use the fact that python will sort arrays (yay)
		def get_sort_key(l):
			sort_key = self.extract_columns(l, keys=keys, dtype=dtype) # extract_columns gives them back tab-sep, but we need to cast them
			sort_key.append(l) # the second element is the line
			return sort_key
		
		for l in self.lines(yieldfinal=True):
			
			# on EOF or read in enough, process
			parts = spaceRegex.split(l)

			if (not l) or len(current_lines) >= lines:
				sort_tmp_path = self.path+".sorted."+str(temp_id)
				o = codecs.open(sort_tmp_path, 'w', ENCODING)
				print >>o, "\n".join(sorted(current_lines, key=get_sort_key))
				o.close()
				
				sorted_tmp_files.append(sort_tmp_path)
				
				current_lines = []
				temp_id += 1
			
			if not l: break
			else: current_lines.append(l)
		
		# okay now go through and merge sort -- use this cool heapq merging trick!
		o = codecs.open(self.path, "w", ENCODING)
		for x in heapq.merge(*map(yield_lines, sorted_tmp_files)):
			print >>o, x[len(x)-1] # the last item is the line itself, everything else is sort keys
		o.close()
		
		# clean up
		for f in sorted_tmp_files: os.remove(f)
		
		if CLEAN_TMP: self.rm_tmp()
		
	def merge(self, other, keys1, tocopy, keys2=None, newheader=None, assert_sorted=True):
		"""
			Copy lines of other that match on keys onto self
			
			other - a LineFile object -- who to merge in
			keys1 - the keys of self for merging
			keys2 - the keys of other for merging. If not specified, we assume they are the same as keys1
			newheader - If specified, gives the names for the *new* columns
			assert_sorted - make False if you don't want an extra check on sorting (things can go very bad)
			
			NOTE: This assumes that every line of self occurs in other, but not vice-versa. It 
			      also allows multiples in self, but *not* other
		"""
		# fix up the keys
		# Note: Keys2 must be processed first here so we can specify by names, 
		#       and not have keys1 overwritten when they are mapped to numbers
		keys2 = listifnot(other.to_column_number(ifelse(keys2 is None, keys1, keys2)))
		tocopy = listifnot(other.to_column_number(tocopy))
		keys1 = listifnot(self.to_column_number(keys1))
		
		# this only works if we are sorted -- let's assert
		if assert_sorted:
			self.assert_sorted(keys1,  allow_equal=True) # we can have repeat lines
			other.assert_sorted(keys2, allow_equal=False) # we cannot have repeat lines (how would they be mapped?)
		
		self.mv_tmp()
		
		o = codecs.open(self.path, "w", ENCODING)
		in1 = codecs.open(self.tmppath, "r", ENCODING)
		in2 = codecs.open(other.path, "r", ENCODING)
		
		line1, parts1, key1 = read_and_parse(in1, keys=keys1)
		line2, parts2, key2 = read_and_parse(in2, keys=keys2)
		
		while True:
			if key1 == key2:
				print >>o, line1+"\t"+"\t".join(self.extract_columns(line2, keys=tocopy))
				
				line1, parts1, key1 = read_and_parse(in1, keys=keys1)
				if not line1: break
			else:
				#print "HERE", key2
				line2, parts2, key2 = read_and_parse(in2, keys=keys2)
				if not line2:  # okay there is no match for line1 anywhere
					print "** Error in merge: end of line2 before end of line 1:"
					print "\t", line1
					print "\t", line2
					exit(1)
		o.close()
		in1.close()
		in2.close()
		
		#update the headers
		self.header.extend([other.header[i] for i in tocopy ]) # copy the header names from other
		
		if CLEAN_TMP: self.rm_tmp()
		
	def print_average_surprisal(self, W, CWcnt, Ccnt, assert_sorted=True):
		"""
			Compute the average in-context surprisal, as in Piantadosi, Tily Gibson (2011)
			
			- W     - column for the word
			- CWcnt - column for the count of context-word
			- Ccnt  - column for the count of the context
			
			NOTE: This prints output rather than 
		"""
		
		W = self.to_column_number(W)
		assert(not isinstance(W,list))
		CWcnt = self.to_column_number(CWcnt)
		assert(not isinstance(CWcnt,list))
		Ccnt = self.to_column_number(Ccnt)
		assert(not isinstance(Ccnt,list))
		
		if assert_sorted:
			self.assert_sorted(listifnot(W),  allow_equal=True)
		
		sumSurprisal = 0
		total_context_count = 0
		total_word_frequency = 0
		prev_w = None
		for parts in self.lines(parts=True, tmp=False):
			
			w = parts[W]
			cwcnt = int(parts[CWcnt])
			ccnt  = int(parts[Ccnt])
			
			if w != prev_w and prev_w is not None:
				# print a bunch of measures
				print "\""+prev_w+"\"", "\t", len(prev_w), "\t", sumSurprisal / total_word_frequency, "\t", log2(total_word_frequency), "\t", total_context_count
				sumSurprisal = 0
				total_word_frequency = 0
			
			prev_w = w
			sumSurprisal -= (log2(cwcnt) - log2(ccnt)) * cwcnt
			total_word_frequency += cwcnt
			total_context_count += 1  # just count how many contexts
		
		# and print at the end
		print prev_w, "\t", len(prev_w), "\t", sumSurprisal / total_word_frequency, "\t", log2(total_word_frequency), "\t", total_context_count
	
	#################################################################################################
	# Iterators
	
	def lines(self, tmp=True, parts=False, yieldfinal=False):
		"""
			Yield me a stripped version of each line of tmplines
			
			- tmp -- do we iterate over path or tmp?
			- parts - if true, we return an array that is split on tabs
			- yieldfinal - give back a final '' 
		"""
		
		if tmp:	inn = codecs.open(self.tmppath, 'r', ENCODING)
		else:   inn = codecs.open(self.path, 'r', ENCODING)
		
		for line in inn:
			line = line.strip()
			if parts: yield spaceRegex.split(line)
			else:     yield line
			
		if yieldfinal: yield ''
		inn.close()
	
	def subsample(self, N=1000000):
		"""
			make me a smaller copy of myself by randomly subsampling
			NOTE: N must fit into memory
		"""
		
		self.mv_tmp()
		
		# We'll use a reservoir sampling algorithm
		sample = []
		
		for idx, line in enumerate(self.lines(tmp=True)):
			if idx < N: 
				sample.append(line)
			else:
				r = random.randrange(idx+1)
				if r < N: sample[r] = line
		
		# now output the sample
		o = codecs.open(self.path, 'w', ENCODING)
		for line in sample: print >>o, line
		o.close()
		
