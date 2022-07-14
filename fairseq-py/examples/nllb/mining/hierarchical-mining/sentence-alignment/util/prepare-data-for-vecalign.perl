#!/usr/bin/perl -w

use strict;

my $SENTENCE_SPLITTER="/private/home/pkoehn/mosesdecoder/scripts/ems/support/split-sentences.perl";

die("syntax: prepare-data-for-vecalign.perl DOCS SENT ISD2_LANGUAGE") unless scalar(@ARGV) == 3;
my ($docs,$sent,$language) = @ARGV;

my $tmp = "$sent.tmp";

open(DOCS,"xzcat $docs.xz |") || die("file not found: $docs.xz");;
while(<DOCS>) {
  chop;
  my ($url_e,$url_f,@DOC_PAIR) = split(/\t/);
  for(my $i=0;$i<2;$i++) {
    my $l   = $i ? $language : "en";
    my $url = $i ? $url_f : $url_e;
    open(TEMP,">>$tmp.txt");
    print TEMP "Rzc2NodXR6IHVuZC9vZGVyIHNvbnN0aW\t$l\t$url\n";
    close(TEMP);
    open(DOC,"| base64 -d | perl -ne 'print \$_.\"\\n\";' | $SENTENCE_SPLITTER -l $l 2>/dev/null | grep -v '^<P>\$' >> $tmp.txt") || die("???");
    print DOC $DOC_PAIR[$i];
    close(DOC);
  }
}
close(DOCS);
