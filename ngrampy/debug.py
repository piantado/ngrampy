""" debug

Utilities for debugging.
Many ideas taken from funcy, http://github.com/Suor/funcy

"""
from __future__ import print_function
import sys
import inspect

def tap(x, end="\n", file=sys.stdout):
    print(x, end=end, file=file)
    return x

def log_calls(fn):
    def _fn(*args, **kwargs):
        binding = inspect.getcallargs(fn, *args, **kwargs)
        binding_str = ", ".join("%s=%s" % item for item in binding.iteritems())
        signature = fn.__name__ + "(%s)" % binding_str
        print(signature, file=sys.stderr)
        return fn(*args, **kwargs)
    return _fn

def myassert(tf, s):
        if not tf:
                print >>sys.stderr, "*** Assertion fail: ",s
        assert tf
