#!/usr/bin/perl -w

use strict;

my $MOSES = "/private/home/pkoehn/mosesdecoder";
my $MOSES_BIN = "$MOSES/bin/moses";
my $SENTENCE_PIECE = "/home/pkoehn/statmt/project/sentencepiece/src";

my ($file,$language,$threads) = @ARGV;
$threads = 1 unless defined($threads);
my $MOSES_MODEL = "/private/home/pkoehn/project/paracrawl/document-alignment/models/fast-$language-en";
my $MOSES_INI = "$MOSES_MODEL//moses.ini";
my $TRUECASE_MODEL = "$MOSES_MODEL/truecase-model.$language";
my $SPM_MODEL = "$MOSES_MODEL/spm.model";

my (@INDEX,%DEDUP);
open(FILE,"xzcat $file|");
# language that requires word segmentation (with sentence piece model)
if (-e $SPM_MODEL) {
  open(DEDUP,"| $SENTENCE_PIECE/spm_encode --model=$SPM_MODEL --output_format=piece | $MOSES/scripts/tokenizer/tokenizer.perl -a -l $language -threads $threads > $file.dedup");
}
# language with truecaser
elsif (-e $TRUECASE_MODEL) {
  open(DEDUP,"| $MOSES/scripts/tokenizer/tokenizer.perl -a -l $language -threads $threads | $MOSES/scripts/recaser/truecase.perl --model $TRUECASE_MODEL > $file.dedup");
}
# language with neither
else {
  open(DEDUP,"| $MOSES/scripts/tokenizer/tokenizer.perl -a -l $language -threads $threads > $file.dedup");
}
while(<FILE>) {
  chop;
  my ($page,$text) = split(/\t/);
  if (defined($DEDUP{$text})) {
    push @INDEX,$DEDUP{$text};
  }
  else {
    print DEDUP "$text\n";
    push @INDEX, scalar keys %DEDUP;
    $DEDUP{$text} = scalar keys %DEDUP;
  }
}
close(DEDUP);
close(FILE);

open(DEDUP,"$file.dedup");
open(SPLIT,">$file.split");
while(<DEDUP>) {
  chop;
  my @WORD = split;
  if (scalar(@WORD)>100) {
    print SPLIT "SPLITSPLITSPLIT\n";
    for(my $i=0;$i<scalar(@WORD)/100;$i++) {
      for(my $j=$i*100;$j<($i+1)*100 && $j<scalar(@WORD);$j++) { 
        print SPLIT " " if $j%100; 
        print SPLIT $WORD[$j];
      } 
      print SPLIT "\n";
    }
    print SPLIT "ENDSPLITENDSPLITENDSPLIT\n";
  }
  else {
    print SPLIT $_."\n";
  }
}
close(SPLIT);
close(DEDUP);

`$MOSES_BIN -f $MOSES_INI --threads $threads < $file.split > $file.dedup.translated 2> $file.dedup.moses.log`;
`rm $file.split`;

my @TRANSLATED;
my $split = 0;
open(TRANSLATED,"$file.dedup.translated");
while(<TRANSLATED>) {
  chop;
  if (/^SPLITSPLITSPLIT/) {
    $split = 1;
    push @TRANSLATED,"";
  }
  elsif (/^ENDSPLITENDSPLITENDSPLIT/) {
    $TRANSLATED[$#TRANSLATED] .= "\n";
    $split = 0;
  }
  elsif ($split) {
    $TRANSLATED[$#TRANSLATED] .= " " unless $TRANSLATED[$#TRANSLATED] eq "";
    $TRANSLATED[$#TRANSLATED] .= $_;
  }
  else {
    push @TRANSLATED,$_."\n";
  }
}
close(TRANSLATED);

foreach (@INDEX) {
  print $TRANSLATED[$_];
}
