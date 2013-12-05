""" 
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This file contains a drop-in replacement for LineFile for in-memory operations.
    It mocks the interface of LineFile, including file-related arguments, 
    but performs all operations in memory. 

    This is not the most efficient way to do this in-memory, but it provides
    compability with scripts written for the on-disk version.

    Where possible, operations on files are replaced with analogous operations 
    on the in-memory data structures. Instead of keeping a file in path and another 
    file in tmppath, main data and temp data are stored in separate lists. But
    these are not always used in the same places that the on-disk version uses
    path and tmppath.

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

	def write(self, it=None, tmp=False, lazy=False):
		if it is None:
			it = self._lines
		if lazy:
			self._lines = it
		else:
			self._lines = list(it)

	def read(self):
		return iter(self._lines)
		
	def copy(self, path=None):
		return deepcopy(self)
	
	def sort(self, keys, lines=None, dtype=unicode, reverse=False):
		"""
			Sort me by my keys.
			
			dtype - the type of the data to be sorted. Should be a castable python type
			        e.g. str, int, float
		"""
		keys = listifnot(self.to_column_number(keys))
		
		# Map a line to sort keys (e.g. respecting dtype, etc) ; we use the fact that python will sort arrays (yay)
		def get_sort_key(l):
			sort_key = self.extract_columns(l, keys=keys, dtype=dtype) # extract_columns gives them back tab-sep, but we need to cast them
			sort_key.append(l) # the second element is the line
			return sort_key

		self.write(sorted(self.lines(), key=get_sort_key))

		
	def __len__(self):
		"""
			How many total lines?
		"""
		return len(list(self._lines))
		
