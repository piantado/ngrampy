#!/usr/bin/perl

# This reads from STDIN and collapses equivalent lines, removes tags, and resums rows while
# discretizing dates. We do this with perl because it is orders of magnitude faster than python,
# largely due to unicode
# NOTE: This will split files up by date for convenience. 

use strict;
use utf8;
# use feature 'unicode_strings';

my $OUT_PATH = $ARGV[0];
my $YEAR_DISCRETE = 50; # Discretize every this many years

my $previous_ngram = "";
my $previous_year = "";

my ($ngram, $year, $cnt, $ccnt);

my %year2out; 
	
while(my $line=<STDIN>) {
	chomp $line;
	$line = lc $line;
	
	$line =~ s/_\w+?(\s+)/$1/g; # Remove _TAGS
	$line =~ s/([a-z])"/$1/g; # remove quotes
	$line =~ s/"([a-z])/$1/g; # remove quotes around words

	if($line =~ m/^(.+)\s([0-9]+)\s([0-9]+)\s([0-9]+)$/){
		$ngram = $1;
		$year = int($2/$YEAR_DISCRETE)*$YEAR_DISCRETE;
		my $this_count = $3;
		my $this_ccnt = $4;
		
# 		print "# $ngram\t$year\t$this_count\t$this_ccnt\n"; # To echo output
		if($ngram ne $previous_ngram or $year != $previous_year) {
			
			if(! exists $year2out{$previous_year}) {
				open($year2out{$previous_year}, ">:utf8", $OUT_PATH.".".$previous_year ) or die "Cannot open!";
				$year2out{$previous_year}->autoflush(1);
				print "# OPENING $OUT_PATH $previous_year\n";
			}
			
			print {$year2out{$previous_year}} "$previous_ngram\t$cnt\t$ccnt\n";
				
			$previous_ngram = $ngram;
			$previous_year  = $year;
			$cnt = $this_count;
			$ccnt = $this_ccnt;
		}
		else{ 
			
			$cnt += $this_count;
			$ccnt += $this_ccnt;
		}
	}

}

print {$year2out{$previous_year}} "$ngram\t$cnt\t$ccnt\n";

foreach my $k(keys %year2out){
	close($year2out{$k});
}
