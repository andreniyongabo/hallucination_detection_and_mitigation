#!/usr/bin/perl -w

use strict;

my ($total,$already,$ok,$fail) = (0,0,0,0);
while(<STDIN>) {
  next if /attempted on invalid dirhandle/;
  print $_ if /^#/;
  next if /^#/;
  chop;
  my @FILE = split(/\t/);
  $FILE[0] =~ /(..)\/([^\/]+)$/ || next;
  my ($subdir,$domain) = ($1,$2);
  `mkdir -p data/$subdir/$domain`;
  shift @FILE;
  foreach my $file_size (@FILE) {
    $total++;
    my ($file,$size)  = split(/ /,$file_size);
    my $local_path = "data/$subdir/$domain/$file";
    if (-e $local_path) {
      my @STAT = stat($local_path);
      if ($STAT[7] == $size) {
	$already++;
        next;
      }
      print "$STAT[7] $size $local_path\n";
    }
    print "scp -o ConnectTimeout=300 -p login.clsp.jhu.edu:statmt/data/site-crawl/data/$subdir/$domain/$file $local_path ".`date`;
    `scp -o ConnectTimeout=300 -p login.clsp.jhu.edu:statmt/data/site-crawl/data/$subdir/$domain/$file $local_path`;
    if (-e $local_path) {
      $ok++;
      $fail=0;
    }
    else {
      $fail++;
      print "sleep(2**$fail)\n";
      sleep(2**$fail);
    }
  }
  print "progress: $ok+$already/$total\n";
}


