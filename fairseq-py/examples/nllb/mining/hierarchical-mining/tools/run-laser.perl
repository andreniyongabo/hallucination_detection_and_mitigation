#!/usr/bin/perl -w

use strict;

die unless scalar(@ARGV) == 6;
my ($FILE_F,$FILE_E,$L_F,$L_E,$OUT,$SIZE,$MODEL) = @ARGV;
$MODEL = "/private/home/schwenk/projects/mlenc/models/laser2a_2b4-2021-04-13.pt" unless defined($MODEL);

`mkdir $OUT.tmp` unless -e "$OUT.tmp";
`split -l $SIZE $FILE_F $OUT.tmp/f.` unless -e "$OUT.tmp/f.aa";
`split -l $SIZE $FILE_E $OUT.tmp/e.` unless -e "$OUT.tmp/e.aa";
open(LS,"ls $OUT.tmp/f.??|");
while(my $file_f = <LS>) {
  chop($file_f);
  my $file_e = $file_f;
  $file_e =~ s/f\.([^\.]+)$/e.$1/;
  my $file_laser = $file_f;
  $file_laser =~ s/f\.([^\.]+)$/laser.$1/;
  next if -e $file_laser;
  `sbatch --gpus-per-node=1 --partition=learnfair --time 60 /private/home/pkoehn/data/paracrawl/tools/run-laser.sh $file_f $file_e $L_F $L_E $file_laser $MODEL`;
}
close(LS);
