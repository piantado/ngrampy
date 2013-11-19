""" 
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This file contains a drop-in replacement for LineFile for in-memory operations.
    It mocks the interface of LineFile, including file-related arguments, 
    but performs all operations in memory. 

    This is not the most efficient way to do this in-memory, but it provides
    compability with scripts written for the on-disk version.

    Where possible, operations on files are replaced with analogous operations 
    on the in-memory data structures. Instead of keeping a file in path and another 
    file in tmppath, main data and temp data are stored in separate lists.

    Richard Futrell, 2013
    
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
import filehandling as fh
from LineFile import LineFile

ENCODING = 'utf-8'
CLEAN_TMP = False
SORT_DEFAULT_LINES = None

# Set this so we can write stderr
#sys.stdout = codecs.getwriter(ENCODING)(sys.stdout)
#sys.stderr = codecs.getwriter(ENCODING)(sys.stderr)

IO_BUFFER_SIZE = int(100e6) # approx size of input buffer 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Class definition
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class LineFileInMemory(LineFile):
	
	def __init__(self, files, header=None, path=None, force_nocopy=False):
		"""
			Create a new file object with the specified header. It takes a list of files
			and reads them into memory as a list of lines.
			
			header - give each column a name (you can refer by name instead of number)
			path   - does nothing, for compatibility with LineFile
			force_nocopy - does nothing, for compatibility with LineFile
		"""
		if isinstance(files, str):
			files = [files]

		self.path = None
		self.tmppath = None

		# load the files into memory
		self._lines = []
		self._tmp_lines = []
		for f in files:
			if f.endswith(".idf"): 
				continue # skip the index files
			else:
				with fh.open(f, encoding=ENCODING) as infile:
					self._lines.extend(infile)

		
		# just keep track
		self.files = files
		
		if isinstance(header, str): 
			self.header = header.split()
		else:
			self.header = header
		
		
	def delete_columns(self, cols):
		
		self.mv_tmp()

		# make sure these are *decreasing* so we can delete in order
		cols = sorted(listifnot(self.to_column_number(cols)), reverse=True)

		def generate():
			for parts in self.lines(parts=True):
				for c in cols: 
					del parts[c]
				yield "\t".join(parts)
			
		self._lines = list(generate())

		# and delete from the header
		if self.header is not None:
			for c in cols: 
				del self.header[c]

		if CLEAN_TMP:
			self.rm_tmp()
		
	def copy(self, path=None):
		raise NotImplementedError
	
	def get_new_path(self):
		raise NotImplementedError
			
	def mv_tmp(self):
		"""
			Move contents of the primarily list of lines to the tmp list of lines.
		"""
		self._tmplines = deepcopy(self._lines)
		del self._lines[:]
		
	def rename(self, n):
		raise NotImplementedError
		
	def rm_tmp(self):
		"""
			Delete the temporary list of lines.
		"""
		del self._tmplines[:]

	def cp(self, f): 
		raise NotImplementedError
	
		
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
		
		def generate(columns=columns):
			toss_count = 0
			total_count = 0
			for l in self.lines():
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
			
		self._lines = list(generate())

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
		def generate():
			for l in self.lines():
				parts = l.split()
				keep = True
				for c in cols: # for each thing to check, check it!
					if (parts[c] in vocabulary) is invert:
						keep = False
						break
				if keep: 
					yield l

		self._lines = list(generate())
		if CLEAN_TMP: 
			self.rm_tmp()
	
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
		
                def generate():
                        for compkey, lines in self.groupby(keys):
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
		
		self._lines = list(generate())
		if CLEAN_TMP: 
			self.rm_tmp()
		
	def cat(self): 
		raise NotImplementedError

	def delete(self):
		del self._lines[:]
		del self._tmplines[:]

	def delete_tmp(self):
		del self._tmplines[:]

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
		
		def generate():
			for line in self.lines():
				parts = line.split()
				yield line+"\t"+function(*[parts[i] for i in keys])
			
			self.header.extend(listifnot(newname))
		
		self._lines = list(generate())

		if CLEAN_TMP: 
			self.rm_tmp()
		
	def sort(self, keys, lines=None, dtype=unicode, reverse=False):
		"""
			Sort me by my keys.
			
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

		self._lines = sorted(self._tmplines, key=get_sort_key)
		
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
		raise NotImplementedError
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
					print >>sys.stderr, "** Error in merge: end of line2 before end of line 1:"
					print >>sys.stderr, "\t", line1
					print >>sys.stderr, "\t", line2
					exit(1)
		o.close()
		in1.close()
		in2.close()
		
		#update the headers
		self.header.extend([other.header[i] for i in tocopy ]) # copy the header names from other
		
		if CLEAN_TMP: self.rm_tmp()
	
			
	#################################################################################################
	# Iterators
	
	def lines(self, tmp=True, parts=False, yieldfinal=False):
		"""
			Yield me a stripped version of each line of tmplines
			
			- tmp -- do we iterate over path or tmp?
			- parts - if true, we return an array that is split on tabs
			- yieldfinal - give back a final '' 
		"""


		if tmp: 
			it = iter(self._tmplines)
		else:
			it = iter(self._lines)

		for line in it:
			line = line.strip()
			if parts: 
				yield line.split()
			else:     
				yield line
			
		if yieldfinal: 
			yield ''

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
		return len(self._lines)
		
	def subsample_lines(self, N=1000000):
		"""
			Make me a smaller copy of myself by randomly subsampling *lines*
			not according to counts. This is useful for creating a temporary
			file 
			NOTE: N must fit into memory
		"""
		raise NotImplementedError
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
		raise NotImplementedError
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


