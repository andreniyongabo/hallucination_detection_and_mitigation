#!/usr/bin/perl -w

use strict;

die("subsample.perl IN_FILE E_POS F_POS SCORE_POS OUT [SIZES]") unless scalar(@ARGV) >= 5;
my ($IN_FILE,$E_POS,$F_POS,$SCORE_POS,$OUT,@SIZE) = @ARGV;

# collect word count per score
my %SCORE;
open(FILE,"xzcat $IN_FILE|") if $IN_FILE =~ /xz$/;
open(FILE,$IN_FILE) unless $IN_FILE =~ /xz$/;
while(my $line = <FILE>) {
  chop($line);
  my @ITEM = split(/\t/,$line);
  my $e     = $ITEM[$E_POS];
  my $score = $ITEM[$SCORE_POS];
  my $e_length = scalar split(/ /,$e);
  $SCORE{$score} += $e_length;
}
close(FILE);

# compute thresholds
my %THRESHOLD;
my $count = 0;
my $lowest_score = 0;
foreach my $score (sort {$b <=> $a} (keys %SCORE)) {
  $count += $SCORE{$score};
  $THRESHOLD{$score} = $count;
  $lowest_score = $score;
}
print "total number of sentences: $count\n";

# find threshold cutoff values for specified sizes
@SIZE = (1e9,2e9,3e9,5e9,7e9) unless scalar(@SIZE);
my %THRESHOLD_CUTOFF;
my %INCLUDED_AT_THRESHOLD;
my $size = 0;
foreach my $score (sort {$b <=> $a} (keys %THRESHOLD)) {
  while ($THRESHOLD{$score} > $SIZE[$size]) { # the first time it goes over threshold...
    print "threshold at $SIZE[$size]: $score\n";
    $INCLUDED_AT_THRESHOLD{$SIZE[$size]} = $SCORE{$score} - ($THRESHOLD{$score} - $SIZE[$size]);
    $THRESHOLD_CUTOFF{$SIZE[$size++]} = $score;
    last if $size == scalar(@SIZE);
  }
  last if $size == scalar(@SIZE);
}
# is that really needed?
foreach my $size (@SIZE) {
  if (!defined($THRESHOLD_CUTOFF{$size})) {
    $THRESHOLD_CUTOFF{$size} = $lowest_score;
    $INCLUDED_AT_THRESHOLD{$size} = $SCORE{$lowest_score} - ($THRESHOLD{$lowest_score} - $size);
  }
}

exit if $OUT eq "no";

# open files to store subsampled sets
my (%OUT_E,%OUT_F);
foreach my $size (@SIZE) {
  open $OUT_E{$size},"> $OUT.$size.e";
  open $OUT_F{$size},"> $OUT.$size.f";
}

# write out sentence pairs scoring over threshold
my %ALREADY_INCLUDED_AT_THRESHOLD;
open(FILE,"xzcat $IN_FILE|") if $IN_FILE =~ /xz$/;
open(FILE,$IN_FILE) unless $IN_FILE =~ /xz$/;
while(my $line = <FILE>) {
  chop($line);
  my @ITEM = split(/\t/,$line);
  my $e     = $ITEM[$E_POS];
  my $f     = $ITEM[$F_POS];
  my $score = $ITEM[$SCORE_POS];
  my $e_length = scalar split(/ /,$e);
  foreach my $size (@SIZE) {
    my $cutoff = $THRESHOLD_CUTOFF{$size};
    next if $score < $cutoff;
    if ($score == $cutoff) {
      next if defined($ALREADY_INCLUDED_AT_THRESHOLD{$size}) &&
        $ALREADY_INCLUDED_AT_THRESHOLD{$size} > $INCLUDED_AT_THRESHOLD{$size};
      my $e_length = scalar split(/ /,$e);
      $ALREADY_INCLUDED_AT_THRESHOLD{$size} += $e_length;
    }
    my $fh_e = $OUT_E{$size};
    my $fh_f = $OUT_F{$size};
    print $fh_e "$e\n";
    print $fh_f "$f\n";
  }
}
close(FILE);

