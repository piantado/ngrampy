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

from debug import *
from helpers import *

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Class definition
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
				else:                       
					systemcall("cat "+f+" >> "+self.path)
		
		# just keep track
		self.files = files
		
		# and store some variables
		self.tmppath = self.path+".tmp"
		
		if isinstance(header, str): 
			self.header = header.split()
		else:
			self.header = header

		self._lazy_lines = None

	def write(self, it, tmp=False, lazy=False):
		if lazy:
			self._lazy_lines = it
		else:
			if tmp:
				filename = self.tmppath
			else:
				filename = self.path

			with codecs.open(filename, 'wb', ENCODING, 
					 'strict', IO_BUFFER_SIZE) as outfile:
				for item in it:
					print >>outfile, item

	def read(self, tmp=False):
		if tmp and self._lazy_lines:
			for item in self._lazy_lines:
				yield item
			self._lazy_lines = None

		if tmp:
			filename = self.tmppath
		else:
			filename = self.path

		with codecs.open(filename, 'rb', ENCODING,
				 'strict', IO_BUFFER_SIZE) as infile:
			for line in infile:
				yield line
		
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
				return map(self.to_column_number, x.split())
			
			# otherwise, a single string so just find the header that equals it
			for i, item in enumerate(self.header):
				if item == x: 
					return i
		
		print >>sys.stderr, "Invalid header name ["+x+"]", self.header
		exit(1)
			
		
	def delete_columns(self, cols):
		
		# make sure these are *decreasing* so we can delete in order
		cols = sorted(listifnot(self.to_column_number(cols)), reverse=True)
		
		self.mv_tmp()
		lines = self.lines(parts=True)
		def generate_deleted(lines=lines):
			for parts in lines:
				for c in cols: 
					del parts[c]
				yield "\t".join(parts)
		
		self.write(generate_deleted())

		# and delete from the header
		if self.header is not None:
			for c in cols: del self.header[c]

		if CLEAN_TMP:
			self.rm_tmp()
		
		
	def copy(self, path=None):
		
		if path is None: path = self.get_new_path() # make a new path if its not specified
		
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
		
		parts = line.split()

		if isinstance(dtype,list):
			return [ dtype[i](parts[x]) for i,x in enumerate(keys)]
		else: 
			return [ dtype(parts[x]) for x in keys ]
			

		
	def clean(self, columns=None, lower=True, alphanumeric=True, count_columns=True, nounderscores=True, echo_toss=False, filter_fn=None, modifier_fn=None):
		"""
			This does several things:
				columns - how many cols should there be? If None, then we use the first line
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

                lines = self.lines()
                def generate_cleaned(columns=columns, lines=lines):
                        toss_count = 0
                        total_count = 0
                        for l in lines: # POINTER TROUBLE!
                                total_count += 1
                                keep = True

                                if filter_fn and not filter_fn(l):
                                        keep = False
                                        if echo_toss:
                                                print >>sys.stderr, "# Tossed filtered line:", l
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
                                if keep:

                                        # clean up according to specs
                                        if nounderscores:
                                                l = re_underscore.sub("", l)
                                        if lower:
                                                l = l.lower()
                                        if modifier_fn:
                                                l = modifier_fn(l)

                                        yield l

                        print >>sys.stderr, "# Clean tossed %i of %i lines, or %s percent" % (toss_count, total_count, str((toss_count/total_count) * 100))

                self.write(generate_cleaned())

                if CLEAN_TMP:
                        self.rm_tmp()


	def restrict_vocabulary(self, cols, vocabulary, invert=False):
		"""
			Make a new version where "cols" contain only words matching the vocabulary
			OR if invert=True, throw out anything matching cols
		"""
		
                cols = listifnot(self.to_column_number(cols))

                vocabulary = set(vocabulary)

                self.mv_tmp()
                lines = self.lines()
                def generate_restricted(lines=lines):
                        for l in lines:
                                parts = l.split()
                                keep = True
                                for c in cols: # for each thing to check, check it!
                                        if (parts[c] in vocabulary) is invert:
                                                keep = False
                                                break
                                if keep:
                                        yield l

                self.write(generate_restricted())

                if CLEAN_TMP:
                        self.rm_tmp()
	
	def make_marginal_column(self, newname, keys, sumkey):
		self.copy_column(newname, sumkey)
		self.sort(keys)
		self.resum_equal(keys, newname, keep_all=True, assert_sorted=False)
	
	def resum_equal(self, keys, sumkeys, assert_sorted=True, keep_all=False):
		"""
			Takes all rows which are equal on the keys and sums the sumkeys, overwriting them. 
			Anything not in keys or sumkeys, there are only guarantees for if keep_all=True.
		"""
                keys    = listifnot(self.to_column_number(keys))
                sumkeys = listifnot(self.to_column_number(sumkeys))

                if assert_sorted: # must call before we mv_tmp
                        self.assert_sorted(keys,  allow_equal=True)

                self.mv_tmp()

                groups = self.groupby(keys)
                def generate_resummed(groups=groups):
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

                self.write(generate_resummed())
                if CLEAN_TMP:
                        self.rm_tmp()		
		
	def assert_sorted(self, keys, dtype=unicode, allow_equal=False):
		"""
			Assert that a file is sorted by certain columns
			This good for merging, etc., which optionally check requirements 
			to be sorted
		"""
		keys = self.to_column_number(keys)
	
		prev_sortkey = None
		for line in self.lines(tmp=False):
			line = line.strip()
			sortkey = self.extract_columns(line, keys=keys, dtype=dtype) # extract_columns gives them back tab-sep, but we need to cast them
			
			if prev_sortkey is not None:
				if allow_equal: 
					myassert( prev_sortkey <= sortkey, line+";"+unicode(prev_sortkey)+";"+unicode(sortkey) )
				else:           
					myassert( prev_sortkey < sortkey, line+";"+unicode(prev_sortkey)+";"+unicode(sortkey) )
			
			prev_sortkey = sortkey
	

	def cat(self): 
		systemcall("cat "+self.path)

        def head(self, n=10):
		print self.header
                lines = self.lines(tmp=False)
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
		os.remove(self.tmppath)

	def copy_column(self, newname, key):
		""" Copy a column. """
		self.make_column(newname, lambda x: x, key)
		
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

                lines = self.lines()
                def generate_new_col(lines=lines):
                        for line in lines:
                                parts = line.split()
                                yield line+"\t"+function(*[parts[i] for i in keys])

                        self.header.extend(listifnot(newname))

                self.write(generate_new_col())

                if CLEAN_TMP:
                        self.rm_tmp()
		
	def sort(self, keys, num_lines=SORT_DEFAULT_LINES, dtype=unicode, reverse=False):
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
		sorted_tmp_files = [] # a list of generators, yielding each line of the file
		
		keys = listifnot(self.to_column_number(keys))

		# a generator to hand back lines of a file and keys for sorting
		def yield_lines(f):
			with codecs.open(f, "r", ENCODING) as infile:
				for l in codecs.open(f, "r", ENCODING): 
					yield get_sort_key(l.strip())
				
		# Map a line to sort keys (e.g. respecting dtype, etc); 
		# we use the fact that python will sort arrays (yay)
		def get_sort_key(l):
			sort_key = self.extract_columns(l, keys=keys, dtype=dtype) 
			sort_key.append(l) # the second element is the line
			return sort_key
		
		for chunk in chunks(self.lines(), num_lines):
			sorted_tmp_path = self.path+".sorted."+str(temp_id)
			with codecs.open(sorted_tmp_path, 'wb', ENCODING) as o:
				print >>o, "\n".join(sorted(chunk, key=get_sort_key))
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
		
		in1 = self.read(tmp=True)
		in2 = other.read(tmp=False)
		
		line1, parts1, key1 = read_and_parse(in1, keys=keys1)
		line2, parts2, key2 = read_and_parse(in2, keys=keys2)

		def generate_merged():
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
		self.write(generate_merged())
		
		#update the headers
		self.header.extend([other.header[i] for i in tocopy ]) # copy the header names from other
		
		if CLEAN_TMP: 
			self.rm_tmp()
	
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
			self.assert_sorted(listifnot(W),  allow_equal=True)
		
		for word, lines in self.groupby(W, tmp=False):
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
	
	def lines(self, tmp=True, parts=False):
		"""
			Yield me a stripped version of each line of tmplines
			
			- tmp -- do we iterate over path or tmp?
			- parts - if true, we return an array that is split on tabs
		"""
		
		inn = self.read(tmp=tmp)

		if parts:
			for line in inn:
				yield line.strip().split()
		else:
			for line in inn:
				yield line.strip()

	def groupby(self, keys, tmp=True):
		"""
                       A groupby iterator matching the given keys.

                """
                keys = listifnot(self.to_column_number(keys))
                key_fn = lambda parts: tuple(parts[x] for x in keys)
                return itertools.groupby(self.lines(parts=True, tmp=tmp), key_fn)
		
	def __len__(self):
		"""
			How many total lines?
		"""
		i = -1
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
		self.write(sample)
	
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
		
		Z = self.sum_column(ccol, tmp=True)
		
		lines = self.lines(parts=True, tmp=True)
		def generate_downsampled(lines=lines):
			for parts in self.lines(parts=True, tmp=True):
				
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
			
		self.write(generate_downsampled())


