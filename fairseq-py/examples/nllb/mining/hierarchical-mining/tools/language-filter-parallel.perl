#!/usr/bin/perl -w

use strict;
use Getopt::Long "GetOptions";

my $FASTTEXT = "/private/home/pkoehn/project/fastText-0.9.2/fasttext";
my $MODEL = "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/model.7.bin";
my $THRESHOLD = 0.9;
my $dir = "/private/home/pkoehn/data/paracrawl/data";

my ($LANGUAGE,$IN);
die("language-filter-parallel.perl [-language LANGUAGE]")
  unless &GetOptions('language=s' => \$LANGUAGE,
                     'in=s' => \$IN);

open(LID_E,"xzcat $IN | cut -f 1 | $FASTTEXT predict-prob $MODEL - 1 0.0 |");
open(LID_F,"xzcat $IN | cut -f 2 | $FASTTEXT predict-prob $MODEL - 1 0.0 |");
open(IN,"xzcat $IN |");
while(my $line = <IN>) {
  my $lid_e = <LID_E>;
  my $lid_f = <LID_F>;
  $lid_e =~ /label__(...) ([\d\.]+)/;
  next unless $1 eq "eng" && $2 > $THRESHOLD;
  $lid_f =~ /label__(...) ([\d\.]+)/;
  next unless $1 eq $LANGUAGE && $2 > $THRESHOLD;
  print $line;
}
close(IN);
close(LID_F);
close(LID_E);
