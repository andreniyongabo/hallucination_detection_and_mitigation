#!/usr/bin/perl -w

use strict;
use MIME::Base64;

my ($docs,$language) = @ARGV;

# english text
my (%EN);
open(EN,"xzcat $docs.en.txt.xz|");
while(<EN>) {
  chop;
  my ($url,$txt) = split(/\t/);
  push @{$EN{$url}}, $txt;
}
close(EN);

# foreign text with translation
my (%F,%FT);
open(F,"xzcat $docs.$language.txt.xz|");
open(FT,"xzcat $docs.$language.translated.xz|");
while(<F>) {
  chop;
  my ($url,$txt) = split(/\t/);
  my $translation = <FT>;
  chop($translation);
  push @{$F{$url}}, $txt;
  push @{$FT{$url}}, $translation;
}
close(FT);
close(F);
  
# load matched URLs and generate input file for bleualign
open(MATCH,"xzcat $docs.matches.xz|");
while(<MATCH>) {
  chop;
  my ($score,$e_url,$f_url) = split(/\t/);
  die("no content for English $e_url") unless defined($EN{$e_url});
  die("no content for foreign $f_url") unless defined($F{$f_url});
  print "$e_url\t$f_url\t";
  print encode_base64(join("\n",@{$F{$f_url}}),"")."\t";
  print encode_base64(join("\n",@{$EN{$e_url}}),"")."\t";
  print encode_base64(join("\n",@{$FT{$f_url}}),"")."\n";
}
close(MATCH);
