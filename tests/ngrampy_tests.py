import os, os.path
import random
import math

from nose.tools import *

from ngrampy.LineFile import LineFile
from ngrampy.LineFileInMemory import LineFileInMemory

try:
    os.mkdir("tests/tmp")
except OSError:
    pass
    

def test_basics():
    G = LineFile("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    assert_equal(G.header, "foo bar baz qux".split())
    assert_equal(G.files, ["tests/smallcorpus.txt.bz2"])
    assert_equal(G.path, "tests/tmp/testcorpus")
    assert_equal(G.tmppath, "tests/tmp/testcorpus.tmp")
    assert_equal(os.path.isfile("tests/tmp/testcorpus"), True)

    G_copy = G.copy()
    copy_path = G_copy.path
    assert_not_equal(copy_path, G.path)

    G_copy.mv_tmp()
    assert_equal(os.path.isfile(G_copy.path + ".tmp"), True)
    G_copy.delete_tmp()
    assert_equal(os.path.isfile(G_copy.path + ".tmp"), False)

    G.make_column("quux", lambda x, y, z, w: "cat", "foo bar baz qux")
    assert_equal(G.header, "foo bar baz qux quux".split())
    for line in G.lines(parts=False, tmp=False):
        assert_equal(G.extract_columns(line, "quux"), ["cat"])
    
    G.delete_columns("quux")
    assert_equal(G.header, "foo bar baz qux".split())

    G.copy_column("quux", "qux")
    assert_equal(G.header, "foo bar baz qux quux".split())
    for line in G.lines(parts=False, tmp=False):
        assert_equal(G.extract_columns(line, "qux"), 
                     G.extract_columns(line, "quux")
                     )

    G.delete()
    assert_equal(os.path.isfile("tests/tmp/testcorpus"), False)

def test_clean():
    G = LineFile("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    len_G = len(G)
    G.clean(columns=4, lower=False, alphanumeric=False, count_columns=True, 
            nounderscores=False, echo_toss=True)
    assert_equal(len(G), len_G - 2)
    G.delete()

    G = LineFile("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=True, count_columns=False, echo_toss=True)
    assert_equal(len(G), 8562)
    G.delete()

    G = LineFile("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=True, count_columns=False, echo_toss=True,
            filter_fn=lambda x: False)
    assert_equal(len(G), 0)
    G.delete()

    G = LineFile("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=False, count_columns=False, echo_toss=True,
            modifier_fn=lambda x: "hello")
    assert_equal(len(G), len_G)
    for line in G.lines(tmp=False, parts=False):
        assert_equal(line, "hello")
    G.delete()


def test_clean_lazy():
    G = LineFile("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    len_G = len(G)
    G.clean(columns=4, lower=False, alphanumeric=False, count_columns=True, 
            nounderscores=False, echo_toss=True, lazy=True)
    assert_equal(len(G), len_G - 2)
    G.delete()

    G = LineFile("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=True, count_columns=False, echo_toss=True, lazy=True)
    assert_equal(len(G), 8562)
    G.delete()

    G = LineFile("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=True, count_columns=False, echo_toss=True,
            filter_fn=lambda x: False, lazy=True)
    assert_equal(len(G), 0)
    G.delete()

    G = LineFile("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=False, count_columns=False, echo_toss=True,
            modifier_fn=lambda x: "hello", lazy=True)
    for line in G.lines(tmp=False, parts=False):
        assert_equal(line, "hello")
    G.delete()
    
def test_resum_equal():
    G = LineFile("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    len_G = len(G)
    total = G.sum_column("qux", tmp=False)
    G.resum_equal("foo", "qux", assert_sorted=True, keep_all=False)
    assert_equal(len(G), 1)
    for line in G.lines(tmp=False):
        assert_equal(int(G.extract_columns(line, "qux")[0]), total)
    G.delete()

    G = LineFile("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.resum_equal("foo", "qux", assert_sorted=True, keep_all=True)
    assert_equal(len(G), len_G)
    for line in G.lines(tmp=False):
        assert_equal(int(G.extract_columns(line, "qux")[0]), total)
    G.delete()

def test_resum_equal_lazy():
    G = LineFile("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    len_G = len(G)
    total = G.sum_column("qux", tmp=False)
    G.resum_equal("foo", "qux", assert_sorted=True, keep_all=False, lazy=True)
    for line in G.lines(tmp=False):
        assert_equal(int(G.extract_columns(line, "qux")[0]), total)
    G.delete()

    G = LineFile("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.resum_equal("foo", "qux", assert_sorted=True, keep_all=True, lazy=True)
    for line in G.lines(tmp=False):
        assert_equal(int(G.extract_columns(line, "qux")[0]), total)
    G.delete()

"""
def test_avg_surprisal():
    G = LineFile("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.make_marginal_column("quux", "foo bar", "qux")
    G.sort("baz")
    for line in G.average_surprisal("baz", "qux", "quux", assert_sorted=True):
        # TODO this test
        pass

    G.delete()
"""

def test_unicode():
    def generate_random_unicode():
        for _ in xrange(5):
            yield unichr(random.choice((0x300, 0x9999)) + random.randint(0, 0xff))

    scramblemap = {}

    G = LineFile("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=False, count_columns=False, echo_toss=True, lazy=True)
    G.make_marginal_column("quux", "foo bar", "qux", lazy=False)
    G.sort("baz")
    len_G = len(G)
    sum_counts = G.sum_column("quux")
    sum_surprisal = math.fsum(line[2] for line in G.average_surprisal("baz", "qux", "quux", assert_sorted=True))
    G.delete()


    G = LineFile("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")

    def scramble(line):
        words = line.split()[:3]
        count = line.split()[-1]
        for i, word in enumerate(words):
            if word in scramblemap:
                words[i] = scramblemap[word]
            else:
                garbage = u"".join(generate_random_unicode())
                words[i] = garbage
                scramblemap[word] = garbage

        return " ".join(words + [count])

    G.clean(lower=True, alphanumeric=False, count_columns=False, echo_toss=True,
            modifier_fn=scramble)
    G.make_marginal_column("quux", "foo bar", "qux")
    G.sort("baz")
    sum_counts_scrambled = G.sum_column("quux")
    assert_equal(sum_counts, sum_counts_scrambled)
    assert_equal(len_G, len(G))
    sum_surprisal_scrambled = math.fsum(line[2] for line in G.average_surprisal("baz", "qux", "quux", assert_sorted=True))
    G.delete()

    assert_equal(sum_surprisal, sum_surprisal_scrambled)

def test_basics_in_memory():
    G = LineFileInMemory("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    assert_equal(G.header, "foo bar baz qux".split())
    assert_equal(G.files, ["tests/smallcorpus.txt.bz2"])

    G.make_column("quux", lambda x, y, z, w: "cat", "foo bar baz qux")
    assert_equal(G.header, "foo bar baz qux quux".split())
    for line in G.lines(parts=False, tmp=False):
        assert_equal(G.extract_columns(line, "quux"), ["cat"])
    
    G.delete_columns("quux")
    assert_equal(G.header, "foo bar baz qux".split())

    G.copy_column("quux", "qux")
    assert_equal(G.header, "foo bar baz qux quux".split())
    for line in G.lines(parts=False, tmp=False):
        assert_equal(G.extract_columns(line, "qux"), 
                     G.extract_columns(line, "quux")
                     )

    G.delete()
    assert_equal(len(G), 0)

def test_clean_in_memory():
    G = LineFileInMemory("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    len_G = len(G)
    G.clean(columns=4, lower=False, alphanumeric=False, count_columns=True, 
            nounderscores=False, echo_toss=True)
    assert_equal(len(G), len_G - 2)
    G.delete()

    G = LineFileInMemory("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=True, count_columns=False, echo_toss=True)
    assert_equal(len(G), 8562)
    G.delete()

    G = LineFileInMemory("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=True, count_columns=False, echo_toss=True,
            filter_fn=lambda x: False)
    assert_equal(len(G), 0)
    G.delete()

    G = LineFileInMemory("tests/smallcorpus-malformed.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=False, count_columns=False, echo_toss=True,
            modifier_fn=lambda x: "hello")
    assert_equal(len(G), len_G)
    for line in G.lines(tmp=False, parts=False):
        assert_equal(line, "hello")
    G.delete()

def test_resum_equal_in_memory():
    G = LineFileInMemory("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    len_G = len(G)
    total = G.sum_column("qux", tmp=False)
    G.resum_equal("foo", "qux", assert_sorted=True, keep_all=False)
    assert_equal(len(G), 1)
    for line in G.lines(tmp=False):
        assert_equal(int(G.extract_columns(line, "qux")[0]), total)
    G.delete()

    G = LineFileInMemory("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.resum_equal("foo", "qux", assert_sorted=True, keep_all=True)
    assert_equal(len(G), len_G)
    for line in G.lines(tmp=False):
        assert_equal(int(G.extract_columns(line, "qux")[0]), total)
    G.delete()

def test_unicode_in_memory():
    def generate_random_unicode():
        for _ in xrange(5):
            yield unichr(random.choice((0x300, 0x9999)) + random.randint(0, 0xff))

    scramblemap = {}

    G = LineFileInMemory("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")
    G.clean(lower=True, alphanumeric=False, count_columns=False, echo_toss=True, lazy=True)
    G.make_marginal_column("quux", "foo bar", "qux", lazy=True)
    G.sort("baz")
    len_G = len(G)
    sum_counts = G.sum_column("quux")
    sum_surprisal = math.fsum(line[2] for line in G.average_surprisal("baz", "qux", "quux", assert_sorted=True))
    G.delete()


    G = LineFileInMemory("tests/smallcorpus.txt.bz2", header="foo bar baz qux", 
                 path="tests/tmp/testcorpus")

    def scramble(line):
        words = line.split()[:3]
        count = line.split()[-1]
        for i, word in enumerate(words):
            if word in scramblemap:
                words[i] = scramblemap[word]
            else:
                garbage = u"".join(generate_random_unicode())
                words[i] = garbage
                scramblemap[word] = garbage

        return " ".join(words + [count])

    G.clean(lower=True, alphanumeric=False, count_columns=False, echo_toss=True,
            modifier_fn=scramble, lazy=True)
    G.make_marginal_column("quux", "foo bar", "qux", lazy=True)
    G.sort("baz")
    sum_counts_scrambled = G.sum_column("quux")
    assert_equal(sum_counts, sum_counts_scrambled)
    assert_equal(len_G, len(G))
    sum_surprisal_scrambled = math.fsum(line[2] for line in G.average_surprisal("baz", "qux", "quux", assert_sorted=True))
    G.delete()

    assert_equal(sum_surprisal, sum_surprisal_scrambled)

