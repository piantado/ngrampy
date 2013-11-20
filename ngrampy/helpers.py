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
non_whitespace_matcher = re.compile(r"[^\s]", re.UNICODE)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Some helpful functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
	it = iter(iterable)
	it_next = it.next
	so_far = []
	so_far_append = so_far.append
	while True:
		try:
			for _ in xrange(size):
				so_far_append(it_next())
			yield so_far
			so_far = []
		except StopIteration:
			if so_far:
				yield so_far
			break

