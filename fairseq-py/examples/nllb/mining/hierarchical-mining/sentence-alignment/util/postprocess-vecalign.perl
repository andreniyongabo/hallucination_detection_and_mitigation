#!/usr/bin/perl -w

use strict;

my ($sent,$language) = @ARGV;
my $tmp = $sent.".tmp";

my %ALIGN;
open(ALIGN,"$tmp.aligned") || die("file not found: $tmp.aligned");;
my $doc_id = -1;
while(<ALIGN>) {
  chop;
  if (/^Rzc2NodXR6IHVuZC9vZGVyIHNvbnN0aW/) {
    $doc_id++;
    next;
  }
  next unless $doc_id>=0;
  s/[\[\]]//g; # remove brackets
  my ($e,$f,$score) = split(/:/);
  next if $e eq "" || $f eq "";
  $score = 0 unless defined($score);
  my @PAIR = ($e,$f,$score);
  push @{$ALIGN{$doc_id}}, \@PAIR;
}
close(ALIGN);

open(DOC,"$tmp.txt") || die("file not found: $tmp.txt");
open(OUT,"|xz - > $sent.xz");
my ($l,$url);
my(%DOC,%DOC_URL);
$doc_id = -1;
while(my $doc_line = <DOC>) {
  chop($doc_line);
  if ($doc_line =~ /^Rzc2NodXR6IHVuZC9vZGVyIHNvbnN0aW/) {
    my $magic;
    ($magic,$l,$url) = split(/\t/,$doc_line);
    if ($l eq "en") {
      &process_one_doc(\%DOC,\%DOC_URL,$ALIGN{$doc_id},$language) if $doc_id>=0 && $l eq "en";
      $doc_id++;
      my (@DUMMY,@DUMMY2);
      $DOC{"en"} = \@DUMMY;
      $DOC{$language} = \@DUMMY2;
    }
    $DOC_URL{$l} = $url;
  }
  else {
    push @{$DOC{$l}}, $doc_line;
  }
}
&process_one_doc(\%DOC,\%DOC_URL,$ALIGN{$doc_id},$language);
close(OUT);
close(DOC);

sub process_one_doc {
  my ($DOC,$DOC_URL,$ALIGN,$language) = @_;
  foreach my $PAIR (@{$ALIGN}) {
    print OUT $$DOC_URL{"en"}."\t";
    print OUT $$DOC_URL{$language}."\t";
    print OUT &combine($$DOC{"en"}, $$PAIR[0])."\t";
    print OUT &combine($$DOC{$language}, $$PAIR[1])."\t";
    print OUT $$PAIR[2]."\n";
  }
}

sub combine {
  my ($SENT,$index) = @_;
  my $out = "";
  foreach my $i (split(/,/,$index)) {
    if ($i > scalar(@{$SENT})) {
      die("index $i out of range ".scalar(@{$SENT}));
    }
    $out .= " ".$$SENT[$i];
  }
  return substr($out,1);
}


