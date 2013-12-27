import sys
import os
import re
from math import log
import subprocess
from itertools import *

ECHO_SYSTEM = False

# Some friendly Regexes. May need to change encoding here for other encodings?
re_SPACE = re.compile(r"\s", re.UNICODE) # for splitting on spaces, etc.
re_underscore = re.compile(r"_[A-Za-z\-\_]+", re.UNICODE) # for filtering out numbers and whitespace
re_collapser  = re.compile(r"[\d\s]", re.UNICODE) # for filtering out numbers and whitespace
re_sentence_boundary = re.compile(r"</?S>", re.UNICODE)
re_tagstartchar = re.compile(r"(\s|^)_", re.UNICODE) # underscores may be okay at the start of words
non_whitespace_matcher = re.compile(r"[^\s]", re.UNICODE)

PRINT_LOG = False # should we log each action?

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Some helpful functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def printlog(x):
	if PRINT_LOG:
		print >>sys.stderr, x
		
def read_and_parse(inn, keys):
                """
                        Read a line and parse it by tabs, returning the line, the tab parts, and some columns
                """
                line = next(inn).strip()
                if not line: 
			return line, None, None
                else:
                        parts = line.split()
                        return line, parts, "\t".join([parts[x] for x in keys])

def systemcall(cmd, echo=ECHO_SYSTEM):
    if echo:
        print >>sys.stderr, cmd
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=None)
    output, _ = p.communicate()
    return output

def ifelse(x, y, z):
        return y if x else z

def listifnot(x):
        return x if isinstance(x, list) else [x]

def log2(x):
        return log(x,2.)

def c2H(counts):
        """ Normalize counts and compute entropy

        Counts can be a generator.
        Doesn't depend on numpy.

        """
        total = 0.0
        clogc = 0.0
        for c in counts:
                total += c
                clogc += c * log(c)
        return -(clogc/total - log(total)) / log(2)

def chunks(iterable, size):
	""" Chunks

	Break an iterable into chunks of specified size.

	Params:
            iterable: An iterable
	    size: An integer size.

	Yields:
            Tuples of size less than or equal to n, chunks of the input iterable.

	Examples:
        >>> lst = ['foo', 'bar', 'baz', 'qux', 'zim', 'cat', 'dog']
        >>> list(chunks(lst, 3))
        [('foo', 'bar', 'baz'), ('qux', 'zim', 'cat'), ('dog',)]

	"""
	it = iter(iterable)
	while True:
		chunk = islice(it, None, size)
		probe = next(chunk) # raises StopIteration if nothing's there
		yield chain([probe], chunk)
