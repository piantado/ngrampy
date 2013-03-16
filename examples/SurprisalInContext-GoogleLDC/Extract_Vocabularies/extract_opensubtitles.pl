#!/usr/bin/perl

# You can retrieve opensubtitles from here: http://opus.lingfil.uu.se/
# Run via: perl extract_opensubtitles.pl en > EnglishVocabulary.txt

# Note: This includes everything, not just real words. 

use strict;
use utf8;

my $language = $ARGV[0]; #"en";
print STDERR "Extracting $language\n";

open(LS, "find /home/piantado/Desktop/mit/Corpora/OpenSubtitles/$language/* | grep '\.gz' | ");
my @files = <LS>;
close(LS);

my %freq;

foreach my $f(@files) {

# 	print STDERR "Processing $language $f\n";
	open(F, "gunzip -c $f | ") or die "Bad file openeing $f";
	my @lines = <F>;
	close(F);
	
	my $txt = join("\n", @lines);
	$txt =~ s/\s+/ /;
	
	while( $txt =~ m/ <w .+?>(.+?)<\/w>/g) {
		my $w = lc $1;
		$w =~ s/[\,\?\!\"\'\:\;\.\-\s]//g; # remove punctuation and spaces that have crept through
		
		# skip a bunch of garbage -- I can't seem to get perl's unicode letter match to do the right thing here.
		if($w =~ m/[0-9\@\#\$\%\^\&\*\(\)\{\}\[\]\\\/\<\>\+\=\`\~]/) { next; }
		if(length($w) == 0) { next; }
		
		# track this as a "word" no matter what
		$freq{$w}++;
	}
}



### Print out in R format:
# print "word\tos.freq\n";
# foreach my $k(sort {$freq{$b} <=> $freq{$a} } keys %freq) {
# 	print "\"$k\"\t$freq{$k}\n";
# }

### Print out just the vocabulary list:
my $n=0;
foreach my $k(sort {$freq{$b} <=> $freq{$a} } keys %freq) {
	print $k."\n";
	$n = $n+1;
}
