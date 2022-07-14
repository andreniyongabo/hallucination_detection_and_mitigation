#!/usr/bin/perl -w

use strict;
use MIME::Base64;

while(<STDIN>) {
  chop;
  my ($url,$txt) = split(/\t/);
  foreach (split(/\n/,decode_base64($txt))) {
    print "$url\t$_\n";
  }
}
