
""" 
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	This class allows manipulation of google ngram data (and similar formatted) data. 
	When you call functions on LineFiles, the changes are echoed in the file. 
	
	The uses tab (\t) as the column separator. 
	
	When you run this, if you get an encoding error, you may need to set the environment to 
	
		export PYTHONIOENCODING=utf-8	
		
	
	TODO: 
		- Make this so each function call etc. will output what it did
		- Make a "separator" and make sure that all the relevant functions use this (instead of space)
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
from __future__ import division
import os
import sys
import re
import unicodedata
import heapq
import shutil
import random
import codecs # for writing utf-8
import itertools
from math import log
from collections import Counter
from copy import deepcopy

try:
	import numpy
except ImportError:
	try:
		import numpypy as numpy
	except ImportError:
		pass

# A temporary file like /tmp
NGRAMPY_DEFAULT_PATH = "/tmp" #If no path is specified, we go here

ECHO_SYSTEM = True # show the system calls we make?
CLEAN_TMP = False # when we create temporary files, do we remove them when we're done? (sorting files are always removed)
SORT_DEFAULT_LINES = 10000000 # how many lines to sorted at a time in RAM when we sort a large file?
ENCODING = 'utf-8'

# Set this so we can write stderr
sys.stdout = codecs.getwriter(ENCODING)(sys.stdout)
sys.stderr = codecs.getwriter(ENCODING)(sys.stderr)

IO_BUFFER_SIZE = int(100e6) # approx size of input buffer 

# Some friendly Regexes. May need to change encoding here for other encodings?
re_SPACE = re.compile(r"\s", re.UNICODE) # for splitting on spaces, etc.
re_underscore = re.compile(r"_[A-Za-z\-\_]+", re.UNICODE) # for filtering out numbers and whitespace
re_collapser  = re.compile(r"[\d\s]", re.UNICODE) # for filtering out numbers and whitespace
non_whitespace_matcher = re.compile(r"[^\s]", re.UNICODE)
		
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
			parts = re_SPACE.split(line)
			return line, parts, "\t".join([parts[x] for x in keys])

def systemcall(x):
	"""
		Call System functions but echo if we want
	"""
	if ECHO_SYSTEM: print >>sys.stderr, x
	os.system(x)

def ifelse(x, y, z):
	return y if x else z
		
def listifnot(x):
	if isinstance(x, list): return x
	else:                   return [x]
	
def myassert(tf, s):
	if not tf: print >>sys.stderr, "*** Assertion fail: ",s
	assert(tf)
	
def log2(x): return log(x,2.)

def c2H(counts):
	# Normalize counts and compute entropy	
	total = 0.0
	clogc = 0.0
	for c in counts:
		total += c
		clogc += c * log2(c)
	return -(clogc/total - log2(total))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Class definition
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class LineFile(object):
	
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
			front_end_filter - a filtering that may remove/collapse some google lines for speed. 
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
		
		if isinstance(header, str): header = re_SPACE.split(header)
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
		
			if re_SPACE.search(x):  # if spaces, treat it as an array and map
				return map(self.to_column_number, re_SPACE.split(x))
			
			# otherwise, a single string so just find the header that equals it
			for i in xrange(len(self.header)):
				if self.header[i] == x: return i
		
		print >>sys.stderr, "Invalid header name ["+x+"]", self.header
		exit(1)
			
		
	def delete_columns(self, cols):
		
		# make sure these are *decreasing* so we can delete in order
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
		#print "# >>", self.path, self.tmppath
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
	
	def extract_columns(self, line, keys, dtype=unicode):
		"""'ut
			Extract some columns from a single line. Assumes that keys are numbers (e.g. already mapped through to_column_number)
			and wil return the columns as the specified dtype
			NOTE: This always returns a list, even if one column is specified. This may change in the future
			
			e.g. line="a\tb\tc\td"
			     keys=[1,4]
			     gives: ["a", "b", "c", "d"], "b\td"
		"""
		if isinstance(keys, str): keys = listifnot(self.to_column_number(keys))
		
		parts = re_SPACE.split(line)
		
		if isinstance(dtype,list):
			return [ dtype[i](parts[x]) for i,x in enumerate(keys)]
		else: 
			return [ dtype(parts[x]) for x in keys ]
			

		
	def clean(self, columns=None, lower=True, alphanumeric=True, count_columns=True, nounderscores=True, echo_toss=False, filter_fn=None, modifier_fn=None):
		"""
			This does several things:
				columns - how many cols should there be? If None, then we use the first line
				tab - Replace all whitspace on each line with tabs
				lower - convert to lowercase
				alphanumeric - toss lines with non-letter category characters (in unicode)
				count_columns - if True, we throw out rows that don't have the same number of columns as the first line
				nounderscores - if True, we remove everything matchin _[^\s]\s -> " "
				echo_toss - tell us who was removed
				already_tabbed - if true, we know to split cols on tabs; otherwise whitespace
				filter_fn - User-provided boolean filtering function
				modifier_fn - User-provided function to modify the line (downcase etc)
		"""
		self.mv_tmp()
		
		toss_count = 0
		total_count = 0
		
		o = codecs.open(self.path, "w", ENCODING)
		
		for l in self.lines():
			total_count += 1
			keep = True

			if filter_fn and not filter_fn(l):
				keep = False
				if echo_toss:
					print >>sys.stderr, "# Tossed non-linguistic line:", l
				toss_count += 1
				continue
			
			if alphanumeric: # if we require alphanumeric
				collapsed = re_collapser.sub("", l) # replace digits and spaces with nothing so allw e have are characters
				for k in collapsed:
					n = unicodedata.category(k)
					if n == "Ll" or n == "Lu": pass
					else: 
						toss_count+=1
						keep = False # throw out lines with non-letter categories
						if echo_toss: 
							print >>sys.stderr, "# Tossed line that was non-alphanumeric:", l
						break
			
			if not keep: 
				continue # we can skip early here if we want

			# clean up according to specs
			if nounderscores: 
				l = re_underscore.sub("", l)			
			if lower: 
				l = l.lower()

			if modifier_fn:
				l = modifier_fn(l)
			
			# check the number of columns
			if count_columns: 
				cols = l.split()
				cn = len(cols)
				if columns is None: 
					columns = cn # save the first line
				
				if columns != cn or any(not non_whitespace_matcher.search(ci) for ci in cols):
					keep = False
					toss_count+=1
					if echo_toss: print >>sys.stderr, "# Tossed line with bad column count (",cn,"):", l
				
			# and print if we should keep it
			#print keep, cn, l.split(), l
			if keep: print >>o, l
			
		o.close()
		
		print >>sys.stderr, "# Clean tossed %i of %i lines, or %s percent" % (toss_count, total_count, str((toss_count/total_count) * 100))
		
		if CLEAN_TMP: self.rm_tmp()
		
	def restrict_vocabulary(self, cols, vocabulary, invert=False):
		"""
			Make a new version where "cols" contain only words matching the vocabulary
			OR if invert=True, throw out anything matching cols
		"""
		
		cols = listifnot(self.to_column_number(cols))
		
		vocabulary = set(vocabulary)
		
		self.mv_tmp()
		o = codecs.open(self.path, "w", ENCODING)
		for l in  self.lines():
			parts = re_SPACE.split(l)
			keep = True
			for c in cols: # for each thing to check, check it!
				if (parts[c] in vocabulary) is invert:
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
		
		keys    = listifnot(self.to_column_number(keys))
		sumkeys = listifnot(self.to_column_number(sumkeys))
		
		if assert_sorted: # must call before we mv_tmp
			self.assert_sorted(keys,  allow_equal=True)
		
		self.mv_tmp()
		
		o = codecs.open(self.path, "w", ENCODING)
		old_compkey = None
		old_parts = None
		sums = defaultdict(int)
		for parts in self.lines(parts=True):

			# what key do we use to compare equality?
			compkey = "\t".join([parts[x] for x in keys])
			
			if compkey == old_compkey: # sum up:
				try:
					for x in sumkeys: sums[x] += int(parts[x])
				except IndexError:
					print "IndexError:", parts, sumkeys
				
			else: # else print out the previous line, if nothing
				
				if old_parts is not None:
					# the easiest way to do this is just to overwrite in parts with the new counts
					for x in sumkeys: old_parts[x] = str(sums[x])
					
					# and print output					
					print >>o, "\t".join(old_parts)
				
				# and update the new 
				sums = defaultdict(int)
				try:
					for x in sumkeys: sums[x] += int(parts[x])
				except IndexError:
					print "IndexError:", parts, sumkeys
					
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
				if allow_equal: 
					myassert( prev_sortkey <= sortkey, line+";"+unicode(prev_sortkey)+";"+unicode(sortkey) )
				else:           
					myassert( prev_sortkey < sortkey, line+";"+unicode(prev_sortkey)+";"+unicode(sortkey) )
			
			prev_sortkey = sortkey
	

	def cat(self): systemcall("cat "+self.path)
	def head(self, n=10): 
		print self.header
		systemcall("head -n "+unicode(n)+" "+self.path)
	def delete(self):
		os.remove(self.path)
		os.remove(self.tmppath)
	def delete_tmp(self):
		os.remove(self.tmppath)

	def copy_column(self, newname, key):
		""" Copy a column. """
		self.make_column(newname, lambda x: x, [key])
		
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
			parts = re_SPACE.split(line)
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
			parts = re_SPACE.split(l)

			if not l or len(current_lines) >= lines:
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
			print >>o, x[-1] # the last item is the line itself, everything else is sort keys
		o.close()
		
		# clean up
		for f in sorted_tmp_files: 
			os.remove(f)
		
		if CLEAN_TMP: 
			self.rm_tmp()
		
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
		keys2 = listifnot(other.to_column_number(keys1 if keys2 is None else keys2))
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
	
	def print_conditional_entropy(self, W, cntXgW, downsample=10000, assert_sorted=True, pre="", preh="", header=True):
		"""
			Print the entropy H[X | W] for each W, assuming sorted by W.
			Here, P(X|W) is given by unnormalized cntXgW
			Also prints the total frequency
			downsample - also prints the downsampled measures, where we only have downsample counts total. An attempt to correct H bias
		"""
		
		if assert_sorted:
			self.assert_sorted(listifnot(W),  allow_equal=True) # allow W to be true
			
		
		W = self.to_column_number(W)
		assert(not isinstance(W,list))
		#Xcol = self.to_column_number(X)
		#assert(not isinstance(X,list))
		
		cntXgW = self.to_column_number(cntXgW)
		assert(not isinstance(cntXgW, list))
		
		if assert_sorted: self.assert_sorted(listifnot(W),  allow_equal=True)
		
		prevW = None
		if header: print preh+"Word\tFrequency\tContextCount\tContextEntropy\tContextEntropy2\tContextEntropy5\tContextEntropy10\tContextEntropy%i\tContextCount%i" % (downsample, downsample)
		for w, lines in self.groupby(W):
			w = w[0] # w comes out as ("hello",)
			wcounts = np.array([float(parts[cntXgW]) for parts in lines])
			sumcount = sum(wcounts)
			dp = numpy.sort(numpy.random.multinomial(downsample, wcounts / sumcount)) # sort so we can take top on next line
			tp2, tp5, tp10 = dp[-2:], dp[-5:], dp[-10:]
			print pre, w, "\t", sumcount, "\t", len(wcounts), "\t", c2H(wcounts), "\t", c2H(tp2), "\t", c2H(tp5), "\t", c2H(tp10), "\t", c2H(dp), "\t", numpy.sum(dp>0)

			
	def print_average_surprisal(self, W, CWcnt, Ccnt, transcribe_fn=None, assert_sorted=True):
		"""
			Compute the average in-context surprisal, as in Piantadosi, Tily Gibson (2011)
			
			- W     - column for the word
			- CWcnt - column for the count of context-word
			- Ccnt  - column for the count of the context
			- transcribe_fn (optional) - transcription to do before measuring word length
			     i.e. convert word to IPA, convert Chinese characters to pinyin, etc.
			
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
		print "Word\tOrthographic.Length\tSurprisal\tLog.Frequency\tTotal.Context.Count"
		for parts in self.lines(parts=True, tmp=False):
			
			w = parts[W]
			cwcnt = int(parts[CWcnt])
			ccnt  = int(parts[Ccnt])
			
			if w != prev_w and prev_w is not None:
				# print a bunch of measures
				print "\""+prev_w+"\"", "\t", len(prev_w), "\t", sumSurprisal / total_word_frequency, "\t", log2(total_word_frequency), "\t", total_context_count
				
				total_context_count = 0
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
		
		if tmp:	inn = codecs.open(self.tmppath, 'r', ENCODING, 'strict', IO_BUFFER_SIZE) # Very much faster than line buffering!
		else:   inn = codecs.open(self.path,    'r', ENCODING, 'strict', IO_BUFFER_SIZE)
		
		#text = inn.readlines(IO_BUFFER_SIZE)
		#while text != []:
			#for line in text:
				#line = line.strip()
				#if parts: yield re_SPACE.split(line)
				#else:     yield line
			#text = inn.readlines(IO_BUFFER_SIZE)
				
		for line in inn:
			line = line.strip()
			if parts: yield re_SPACE.split(line)
			else:     yield line
			
		if yieldfinal: yield ''
		inn.close()
		
	def __len__(self):
		"""
			How many total lines?
		"""
		for i, _ in enumerate(self.lines(tmp=False)):
			pass
		return i+1
		
	def subsample_lines(self, N=1000000):
		"""
			Make me a smaller copy of myself by randomly subsampling *lines*
			not according to counts. This is useful for creating a temporary
			file 
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
	
	def sum_column(self, col, cast=int, tmp=True):
		
		col = self.to_column_number(col)
		return sum(cast(parts[col]) for parts in self.lines(parts=True, tmp=tmp))
		
	def downsample_tokens(self, N, ccol, keep_zero_counts=False):
		"""
			Subsample myself via counts with the existing probability distribution.
			- N - the total sample size we end up with.
			- ccol - the column we use to estimate probabilities. Unnormalized, non-log probs (e.g. counts)
			
			NOTE: this assumes a multinomial on trigrams, which may not be accurate. If you started from a corpus, this will NOT in general keep
			counts consistent with a corpus. 
			
			This uses a conditional beta distribution, once for each line for a total of N.
			See pg 12 of w3.jouy.inra.fr/unites/miaj/public/nosdoc/rap2012-5.pdf
		"""
		self.header.extend(ccol)
		ccol = self.to_column_number(ccol)
		
		self.mv_tmp()
		
		Z = self.sum_column(ccol, tmp=True) ## TODO: CREATE SUM_COLUMN, giving the normalizer for the probability (counts)
		
		o = codecs.open(self.path, "w", ENCODING)
		for parts in self.lines(parts=True, tmp=True):
			
			cnt = int(parts[ccol]) 
			
			# Randomly sample
			if N > 0: newcnt = numpy.random.binomial(N,float(cnt)/float(Z))
			else:     newcnt = 0
			
			# Update the conditional multinomial
			N = N-newcnt # samples to draw
			Z = Z-cnt    # normalizer for everything else
			
			parts[ccol] = str(newcnt) # update this
			
			if keep_zero_counts or newcnt > 0:
				print >>o, '\t'.join(parts)
			
		o.close()


