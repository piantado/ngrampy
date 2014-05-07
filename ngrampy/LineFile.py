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
import codecs
import itertools
from math import log
from collections import Counter
from copy import deepcopy

# handly numpy with pypy
try:
	import numpy
except ImportError:
	try:
		import numpypy as numpy
	except ImportError:
		pass

from debug import *
from helpers import *

# A temporary file like /tmp
NGRAMPY_DEFAULT_PATH = "/tmp" #If no path is specified, we go here

ECHO_SYSTEM = True # show the system calls we make?
SORT_DEFAULT_LINES = 10000000 # how many lines to sorted at a time in RAM when we sort a large file?
ENCODING = 'utf-8'

# Set this so we can write stderr
sys.stdout = codecs.getwriter(ENCODING)(sys.stdout)
sys.stderr = codecs.getwriter(ENCODING)(sys.stderr)

IO_BUFFER_SIZE = int(100e6) # approx size of input buffer 

COLUMN_SEPARATOR = u"\t" # must be passed to string.split() or else empty columns are collapsed!

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Class definition
# # # # # # # # # # # # # # # # # s# # # # # # # # # # # # # # # # # # # # # # # # # # #

class LineFile(object):
	
	def __init__(self, files, header=None, path=None, force_nocopy=False):
		"""
			Create a new file object with the specified header. It takes a list of files
			and cats them to path (overwriting it). A single file is acceptable.
			
			header - give each column a name (you can refer by name instead of number)
			path   - where is this file stored? If None, we make a new temporary files
			force_nocopy - Primarily for debugging, this prevents us from copying a file and just uses the path as is
					you should pass files a list of length 1 which is the file, and path should be None
					as in, LineFile(files=["/ssd/trigram-stats"], header="w1 w2 w3 c123 c1 c2 c3 c12 c23 unigram bigram trigram", force_nocopy=True)
		"""
		if isinstance(files, str):
			files = [files]

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
				if f.endswith(".idf"): 
					continue # skip the index files
				if f.endswith(".gz"):    
					systemcall("gunzip -d -c "+f+" >> "+self.path)
				elif f.endswith(".bz2"): 
					systemcall("bzip2 -d -c "+f+" >> "+self.path)
				elif f.endswith(".xz") or f.endswith(".lzma"):
					systemcall("xz -d -c "+f+" >> "+self.path)
				else:                       
					systemcall("cat "+f+" >> "+self.path)
		
		# just keep track
		self.files = files
		
		# and store some variables
		self.tmppath = self.path+".tmp"
		
		if isinstance(header, str): 
			self.header = header.split(COLUMN_SEPARATOR)
		else:
			self.header = header

		self._lines = None
		self.preprocess()

	def preprocess(self):
		def fix_separators(line):
			return COLUMN_SEPARATOR.join(line.split())
		self.map(fix_separators)

	def write(self, it, lazy=False):
		""" Write

		Write the lines in an iterable to the LineFile.

		If lazy, then delay actually evaluating the iterable and
		writing it to file. 

		WARNING! If you specify lazy=True, then you can only read()
		those lines once! If you need to read lines more than once,
		you need to do lazy=False and write the lines to the file. 

		Lazy iterators can be chained into efficient pipelines.

		"""
		if lazy:
			self._lines = it
		else:
			# Write lines to tmppath (note it used to be the other way around!)
			with codecs.open(self.tmppath, mode='w', encoding=ENCODING, 
				 errors='strict', buffering=IO_BUFFER_SIZE) as outfile:
				for item in it:
					print >>outfile, item

			# And move tmppath to path
			self.mv_from_tmp()

	def read(self):
		""" Read

		Return the current lines of the LineFile, whether from
		a file or from a lazy iterator.

		"""
		if self._lines is None:
			return codecs.open(self.path, mode='r', encoding=ENCODING,
					   errors='strict', buffering=IO_BUFFER_SIZE)
		else:
			result = iter(self._lines)
			self._lines = None # only allow the lazy iterator to be read once!!
			return result

		
	def setheader(self, *x): 
		self.header = x

	def rename_column(self, x, v): 
		self.header[self.to_column_number(x)] = v
		
	def to_column_number(self, x):
		"""
		 Takes either:
			a column number - just echoes back
			a string - returns the right column number for the string
			a whitespace separated string - returns an array of column numbers
			an array - maps along and returns
		 
		"""		
		if isinstance(x, int):    
			return x
		elif isinstance(x, list): 
			return map(self.to_column_number, x)
		elif isinstance(x, str): 
			if re_SPACE.search(x):  # if spaces, treat it as an array and map
				return map(self.to_column_number, x.split(" "))
			
			# otherwise, a single string so just find the header that equals it
			for i, item in enumerate(self.header):
				if item == x: 
					return i
		
		print >>sys.stderr, "Invalid header name ["+x+"]", self.header
		exit(1)
			
	def delete_columns(self, cols, lazy=False):
		
		# make sure these are *decreasing* so we can delete in order
		cols = sorted(listifnot(self.to_column_number(cols)), reverse=True)
		
		def generate_deleted(lines):
			for parts in lines:
				for c in cols: 
					del parts[c]
				yield "\t".join(parts)
			# and delete from the header, after deletion is complete
			if self.header is not None:
				for c in cols: 
					del self.header[c]
		
		self.write(generate_deleted(self.lines(parts=True)), lazy=lazy)


		
	def copy(self, path=None):
		
		if path is None: 
			   path = self.get_new_path() # make a new path if its not specified
		
		# we can just copy the file by treating it as one of the "files"
		# and then use this new path, not the old one!
		return LineFile([self.path], header=deepcopy(self.header), path=path)
	
	def get_new_path(self): 
		ind = 1
		while True:
			path = os.path.dirname(self.path)+"/ngrampy-"+str(ind)
			if not os.path.isfile(path): 
				return path
			ind += 1
			
	def mv_tmp(self):
		"""
			Move myself to my temporary file, so that I can cat to my self.path
		"""
		#print "# >>", self.path, self.tmppath
		shutil.move(self.path, self.tmppath)

	def mv_from_tmp(self):
		"""
		        Move myself from self.tmppath to self.path.
		"""
		shutil.move(self.tmppath, self.path)
			
		
	def rename(self, n):
		shutil.move(self.path, n)
		self.path = n
		
	def rm_tmp(self):
		"""
			Remove the temporary file
		"""
		os.remove(self.tmppath)

	def cp(self, f): 
		shutil.cp(self.path, f)
	
	def extract_columns(self, line, keys, dtype=unicode):
		"""
			Extract some columns from a single line. Assumes that keys are numbers (e.g. already mapped through to_column_number)
			and will return the columns as the specified dtype
			NOTE: This always returns a list, even if one column is specified. This may change in the future
			
			e.g. line="a\tb\tc\td"
			     keys=[1,4]
			     gives: ["a", "b", "c", "d"], "b\td"
		"""
		if isinstance(keys, str): 
			keys = listifnot(self.to_column_number(keys))
		
		parts = line.split(COLUMN_SEPARATOR)

		if isinstance(dtype,list):
			return [ dtype[i](parts[x]) for i,x in enumerate(keys)]
		else: 
			return [ dtype(parts[x]) for x in keys ]

	def filter(self, fn, lazy=False, verbose=False):
		""" Keep only lines where the function returns True. """
		if verbose:
			def echo_wrapper(fn):
				def wrapper(x, **kwargs):
					result = fn(x, **kwargs)
					if not result:
						print >>sys.stderr, u"Tossed line due to %s:" % fn.__name__, x
					return result
				return wrapper
			fn = echo_wrapper(fn)

		filtered = itertools.ifilter(fn, self.lines())
		self.write(filtered, lazy=lazy)

	def map(self, fn, lazy=False, verbose=False):
		""" Apply function to all lines. """
		if verbose:
			def echo_wrapper(fn):
				def wrapper(x, **kwargs):
					result = fn(x, **kwargs)
					print >>sys.stderr, u"%s => %s" % (unicode(x), unicode(result))
					return result
				return wrapper
			fn = echo_wrapper(fn)

		mapped = itertools.imap(fn, self.lines())
		self.write(mapped, lazy=lazy)
		
	def clean(self, columns=None, lower=True, alphanumeric=True, count_columns=True, nounderscores=True, echo_toss=False, filter_fn=None, modifier_fn=None, lazy=False):
		"""
			This does several things:
				columns - how many cols should there be? If None, then we use the first line
				lower - convert to lowercase
				alphanumeric - toss lines with non-letter category characters (in unicode). WARNING: Tosses lines with "_" (e.g. syntactic tags in google)
				count_columns - if True, we throw out rows that don't have the same number of columns as the first line
				nounderscores - if True, we remove everything matching _[^\s]\s -> " " 
				echo_toss - tell us who was removed
				filter_fn - User-provided boolean filtering function
				modifier_fn - User-provided function to modify the line (downcase etc)
				
			NOTE: filtering by alphanumeric allows underscores at the beginning of columns (as in google tags)
			NOTE: nounderscores may remove columns if there is a column for tags (e.g. a column with _adv)
		"""
		def filter_alphanumeric(line):
			collapsed = re_tagstartchar.sub("", line) # remove these so that tags don't cause us to toss lines. Must come before spaces removed
			collapsed = re_collapser.sub("", collapsed)
			collapsed = re_sentence_boundary.sub("", collapsed)
			char_categories = (unicodedata.category(k) for k in collapsed)
			return all(n == "Ll" or n == "Lu" for n in char_categories)

		def generate_filtered_columns(lines, columns=columns):
			for line in lines:
				cols = line.split(COLUMN_SEPARATOR)
				cn = len(cols)
				if columns is None:
					columns = cn # save the first line

				if not (columns != cn or any(not non_whitespace_matcher.search(ci) for ci in cols)):
					yield line
				elif echo_toss:
					print >>sys.stderr, "Tossed line with bad column count: %s" % line
					print >>sys.stderr, "Line has %d columns; I expected %d." % (cn, columns)

		# Filters.
		if filter_fn:
			self.filter(filter_fn, lazy=True, verbose=echo_toss)
		if alphanumeric:
			self.filter(filter_alphanumeric, lazy=True, verbose=echo_toss)
		if count_columns:
			self.write(generate_filtered_columns(self.lines()), lazy=True)

		# Maps.
		if nounderscores:
			self.map(lambda line: re_underscore.sub("", line), lazy=True)
		if lower:
			self.map(lambda line: line.lower(), lazy=True)
		if modifier_fn:
			self.map(modifier_fn, lazy=True)
		
		if not lazy:
			self.write(self.lines())

	def restrict_vocabulary(self, cols, vocabulary, invert=False, lazy=False):
		"""
			Make a new version where "cols" contain only words matching the vocabulary
			OR if invert=True, throw out anything matching cols
		"""
		
                cols = listifnot(self.to_column_number(cols))

                vocabulary = set(vocabulary)

		def restrict(line, cols=cols, vocabulary=vocabulary):
			parts = line.split(COLUMN_SEPARATOR)
			for c in cols:
				if invert and parts[c] not in vocabulary:
					return l
				elif parts[c] in vocabulary:
					return l

		self.map(restrict, lazy=lazy)

	def make_marginal_column(self, newname, keys, sumkey, lazy=False):
		self.copy_column(newname, sumkey, lazy=True)
		self.sort(keys)
		self.resum_equal(keys, newname, keep_all=True, assert_sorted=False, lazy=lazy)
	
	def resum_equal(self, keys, sumkeys, assert_sorted=True, keep_all=False, lazy=False):
		"""
			Takes all rows which are equal on the keys and sums the sumkeys, overwriting them. 
			Anything not in keys or sumkeys, there are only guarantees for if keep_all=True.
		"""
                keys    = listifnot(self.to_column_number(keys))
                sumkeys = listifnot(self.to_column_number(sumkeys))

                if assert_sorted: 
                        self.assert_sorted(keys, allow_equal=True, lazy=True)

                def generate_resummed(groups):
                        for compkey, lines in groups:
                                if keep_all:
                                        lines = list(lines) # load into memory; otherwise we can only iterate through once
                                sums = Counter()
                                for parts in lines:
                                        for sumkey in sumkeys:
                                                try:
                                                        sums[sumkey] += int(parts[sumkey])
                                                except IndexError:
                                                        print >>sys.stderr, "IndexError:", parts, sumkeys
                                if keep_all:
                                        for parts in lines:
                                                for sumkey in sumkeys:
                                                        parts[sumkey] = str(sums[sumkey])
                                                yield "\t".join(parts)
                                else:
                                        for sumkey in sumkeys:
                                                parts[sumkey] = str(sums[sumkey]) # "parts" is the last line
                                        yield "\t".join(parts)

		groups = self.groupby(keys)
		self.write(generate_resummed(groups), lazy=lazy)
		
	def assert_sorted(self, keys, dtype=unicode, allow_equal=False, lazy=False):
		"""
			Assert that a file is sorted by certain columns
			This good for merging, etc., which optionally check requirements 
			to be sorted

		"""
		def gen_assert_sorted(lines, keys=keys):
			""" yield lines while asserted their sortedness """
			keys = self.to_column_number(keys)
			prev_sortkey = None
			for line in lines:
				line = line.strip()
				yield line # yield all line and check afterwards
				sortkey = self.extract_columns(line, keys=keys, dtype=dtype)
			
				if prev_sortkey is not None:
					if allow_equal: 
						myassert(prev_sortkey <= sortkey, line+";"+unicode(prev_sortkey)+";"+unicode(sortkey))
				else:           
					myassert(prev_sortkey < sortkey, line+";"+unicode(prev_sortkey)+";"+unicode(sortkey))
			
			prev_sortkey = sortkey
			
		self.write(gen_assert_sorted(self.lines()), lazy=lazy)
	
	def cat(self): 
		systemcall("cat "+self.path)

        def head(self, n=10):
		print self.header
                lines = self.lines()
                for _ in xrange(n):
                        print next(lines)

	def delete(self):
		try:
			os.remove(self.path)
		except OSError:
			pass
		try:
			os.remove(self.tmppath)
		except OSError:
			pass # no temporary file exists

	def delete_tmp(self):
		print >>sys.stderr, "*** delete_tmp now phased out! Please remove from code!"
		#os.remove(self.tmppath)

	def copy_column(self, newname, key, lazy=False):
		""" Copy a column. """
                key = self.to_column_number(key)

		def generate_new_col(lines):
			for line in lines:
				parts = line.split(COLUMN_SEPARATOR)
				yield "\t".join([line, parts[key]])
			self.header.extend(listifnot(newname))

		self.write(generate_new_col(self.lines()), lazy=lazy)

	def make_column(self, newname, function, keys, lazy=False):
		"""
			Make a new column as some function of the other rows
			make_column("unigram", lambda x,y: int(x)+int(y), "cnt1 cnt2")
			will make a column called "unigram" that is the sum of cnt1 cnt2
			
			NOTE: The function MUST take strings and return strings, or else we die
			
			newname - the name for the new column. You can pass multiple if function returns tab-sep strings
			function - a function of other row arguments. Must return strings
			args - column names to get the arguments
		"""
                keys = listifnot( self.to_column_number(keys) )

		def generate_new_col(lines):
			for line in lines:
				parts = line.split(COLUMN_SEPARATOR)
				yield "\t".join([line, function(*[parts[i] for i in keys])])
			self.header.extend(listifnot(newname))

		self.write(generate_new_col(self.lines()), lazy=lazy)

	def sort(self, keys, num_lines=SORT_DEFAULT_LINES, dtype=unicode, reverse=False):
		"""
			Sort me by my keys. this breaks the file up into subfiles of "lines", sorts them in RAM, 
			and the mergesorts them
			
			We could use unix "sort" but that gives weirdness sometimes, and doesn't handle our keys
			as nicely, since it treats spaces in a counterintuitive way
			
			dtype - the type of the data to be sorted. Should be a castable python type
			        e.g. str, int, float
		"""
		sorted_tmp_files = [] # a list of generators, yielding each line of the file
		
		keys = listifnot(self.to_column_number(keys))

		# a generator to hand back lines of a file and keys for sorting
		def yield_lines(f):
			with codecs.open(f, "r", encoding=ENCODING) as infile:
				for l in infile:
					yield get_sort_key(l.strip())
				
		# Map a line to sort keys (e.g. respecting dtype, etc); 
		# we use the fact that python will sort arrays (yay)
		def get_sort_key(l):
			sort_key = self.extract_columns(l, keys=keys, dtype=dtype) 
			sort_key.append(l) # the second element is the line
			return sort_key
		
		temp_id = 0
		for chunk in chunks(self.lines(), num_lines):
			sorted_tmp_path = self.path+".sorted."+str(temp_id)
			with codecs.open(sorted_tmp_path, 'w', encoding=ENCODING) as outfile:
				print >>outfile, "\n".join(sorted(chunk, key=get_sort_key))
			sorted_tmp_files.append(sorted_tmp_path)
			temp_id += 1
		
		# okay now go through and merge sort -- use this cool heapq merging trick!
		def merge_sort():
			for x in heapq.merge(*map(yield_lines, sorted_tmp_files)):
				yield x[-1] # the last item is the line itself, everything else is sort keys

		self.write(merge_sort())
		
		# clean up
		for f in sorted_tmp_files: 
			os.remove(f)
		
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
			self.assert_sorted(keys1,  allow_equal=True, lazy=True) # we can have repeat lines
			other.assert_sorted(keys2, allow_equal=False, lazy=True) # we cannot have repeat lines (how would they be mapped?)
		
		in1 = self.lines()
		in2 = other.lines()
		
		line1, parts1, key1 = read_and_parse(in1, keys=keys1)
		line2, parts2, key2 = read_and_parse(in2, keys=keys2)

		def generate_merged(in1, in2):
			while True:
				if key1 == key2:
					yield line1+"\t"+"\t".join(self.extract_columns(line2, keys=tocopy))
					
					line1, parts1, key1 = read_and_parse(in1, keys=keys1)
					if not line1: 
						break
				else:
					line2, parts2, key2 = read_and_parse(in2, keys=keys2)
					if not line2:  # okay there is no match for line1 anywhere
						print >>sys.stderr, "** Error in merge: end of line2 before end of line 1:"
						print >>sys.stderr, "\t", line1
						print >>sys.stderr, "\t", line2
						exit(1)
			self.header.extend([other.header[i] for i in tocopy ]) # copy the header names from other

		self.write(generate_merged(in1, in2))

	def print_conditional_entropy(self, W, cntXgW, downsample=10000, assert_sorted=True, pre="", preh="", header=True):
		"""
			Print the entropy H[X | W] for each W, assuming sorted by W.
			Here, P(X|W) is given by unnormalized cntXgW
			Also prints the total frequency
			downsample - also prints the downsampled measures, where we only have downsample counts total. An attempt to correct H bias
		"""
		if assert_sorted:
			self.assert_sorted(listifnot(W),  allow_equal=True, lazy=True) # allow W to be true
			
		
		W = self.to_column_number(W)
		assert not isinstance(W,list)
		#Xcol = self.to_column_number(X)
		#assert not isinstance(X,list)
		
		cntXgW = self.to_column_number(cntXgW)
		assert not isinstance(cntXgW, list)
		
		prevW = None
		if header: print preh+"Word\tFrequency\tContextCount\tContextEntropy\tContextEntropy2\tContextEntropy5\tContextEntropy10\tContextEntropy%i\tContextCount%i" % (downsample, downsample)
		for w, lines in self.groupby(W):
			w = w[0] # w comes out as ("hello",)
			wcounts = np.array([float(parts[cntXgW]) for parts in lines])
			sumcount = sum(wcounts)
			dp = numpy.sort(numpy.random.multinomial(downsample, wcounts / sumcount)) # sort so we can take top on next line
			tp2, tp5, tp10 = dp[-2:], dp[-5:], dp[-10:]
			print pre, w, "\t", sumcount, "\t", len(wcounts), "\t", c2H(wcounts), "\t", c2H(tp2), "\t", c2H(tp5), "\t", c2H(tp10), "\t", c2H(dp), "\t", numpy.sum(dp>0)

			
	def average_surprisal(self, W, CWcnt, Ccnt, transcribe_fn=None, assert_sorted=True):
		"""
			Compute the average in-context surprisal, as in Piantadosi, Tily Gibson (2011). 
			Yield output for each word.
			
			- W     - column for the word
			- CWcnt - column for the count of context-word
			- Ccnt  - column for the count of the context
			- transcribe_fn (optional) - transcription to do before measuring word length
			     i.e. convert word to IPA, convert Chinese characters to pinyin, etc.
			
		"""
		
		W = self.to_column_number(W)
		assert(not isinstance(W,list))
		CWcnt = self.to_column_number(CWcnt)
		assert(not isinstance(CWcnt,list))
		Ccnt = self.to_column_number(Ccnt)
		assert(not isinstance(Ccnt,list))
		
		if assert_sorted:
			self.assert_sorted(listifnot(W), allow_equal=True, lazy=True)
		
		for word, lines in self.groupby(W):
			word = word[0] # word comes out as (word,)
			if transcribe_fn:
				word = transcribe_fn(word)
			sum_surprisal = 0
			total_word_frequency = 0
			total_context_count = 0
			for parts in lines:
				cwcnt = int(parts[CWcnt])
				ccnt = int(parts[Ccnt])
				sum_surprisal -= (log2(cwcnt) - log2(ccnt)) * cwcnt
				total_word_frequency += cwcnt
				total_context_count += 1
				length = len(word)
			yield u'"%s"'%word, length, sum_surprisal/total_word_frequency, log2(total_word_frequency), total_context_count

	def print_average_surprisal(self, W, CWcnt, Ccnt, transcribe_fn=None, assert_sorted=True):
		print "Word\tOrthographic.Length\tSurprisal\tLog.Frequency\tTotal.Context.Count"
		for line in self.average_surprisal(W, CWcnt, Ccnt, 
			       transcribe_fn=transcribe_fn, assert_sorted=assert_sorted):
			print u"\t".join(map(unicode, line))
	
	#################################################################################################
	# Iterators
	
	def lines(self, parts=False):
		"""
			Yield me a stripped version of each line of tmplines
			
			- parts - if true, we return an array that is split on tabs

		"""
		if parts:
			return (line.strip().split(COLUMN_SEPARATOR) for line in self.read())
		else:
			return (line.strip() for line in self.read())

	def groupby(self, keys):
		"""
                       A groupby iterator matching the given keys.

                """
                keys = listifnot(self.to_column_number(keys))
                key_fn = lambda parts: tuple(parts[x] for x in keys)
                return itertools.groupby(self.lines(parts=True), key_fn)
		
	def __len__(self):
		"""
			How many total lines?
		"""
		return sum(1 for _ in self.read())
		
	def subsample_lines(self, N=1000000):
		"""
			Make me a smaller copy of myself by randomly subsampling *lines*
			not according to counts. This is useful for creating a temporary
			file 
			NOTE: N must fit into memory
		"""
		
		# We'll use a reservoir sampling algorithm
		sample = []
		
		for idx, line in enumerate(self.lines()):
			if idx < N: 
				sample.append(line)
			else:
				r = random.randrange(idx+1)
				if r < N: sample[r] = line
		
		# now output the sample
		self.write(sample)
	
	def sum_column(self, col, cast=int):		
		col = self.to_column_number(col)
		return sum(cast(parts[col]) for parts in self.lines(parts=True))
		
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
		
		Z = self.sum_column(ccol)
		
		def generate_downsampled(lines):
			for parts in lines:
				
				cnt = int(parts[ccol]) 
			
				# Randomly sample
				if N > 0: 
					newcnt = numpy.random.binomial(N,float(cnt)/float(Z))
				else:     
					newcnt = 0
			
				# Update the conditional multinomial
				N = N-newcnt # samples to draw
				Z = Z-cnt    # normalizer for everything else
			
				parts[ccol] = str(newcnt) # update this
				
				if keep_zero_counts or newcnt > 0:
					yield '\t'.join(parts)

		self.write(generate_downsampled(self.lines(parts=True)))
