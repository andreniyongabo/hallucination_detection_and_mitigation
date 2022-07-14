#!/usr/bin/perl -w

use strict;
use MIME::Base64;
use IPC::Open3;
use Symbol "gensym";

$|=1;

my ($splitter) = @ARGV;

my($writer, $reader);
my $err = gensym;
my $pid = open3($writer, $reader, $err, $splitter);

while(<STDIN>) {
  my $doc = $_;
  my $doc_processed = "";
  foreach my $line (split(/\n/,decode_base64($doc))) {
	  #print STDERR "SPLIT LINE IN $line\n";
    next if $line =~ /^\s*$/;
    $line =~ s/^</&lt;/;
    print $writer $line."\n\n";
    while(my $line_split = <$reader>) {
	    #print STDERR "SPLIT LINE OUT $line_split";
      next if $line_split =~ /^\s*$/;
      last if $line_split =~ /^<P>$/;
      $line_split =~ s/&lt;/</;
      $doc_processed .= $line_split;
    }
  }
  print encode_base64($doc_processed,"")."\n";
}

