#!/usr/bin/perl -w

use strict;
use MIME::Base64;
use IPC::Open3;
use Symbol "gensym";

$|=1;

my ($tokenizer) = @ARGV;

my($writer, $reader);
my $err = gensym;
my $pid = open3($writer, $reader, $err, $tokenizer);

while(<STDIN>) {
  my $doc = $_;
  my $doc_processed = "";
  foreach my $line (split(/\n/,decode_base64($doc))) {
    print STDERR "TOK LINE IN $line\n";
    print $writer $line."\n";
    my $line_tok = <$reader>;
    print STDERR "TOK LINE OUT $line_tok\n";
    $doc_processed .= $line_tok;
  }
  print encode_base64($doc_processed,"")."\n";
}

