#!/usr/bin/perl -w

use strict;
use Getopt::Long "GetOptions";
use MIME::Base64;

my $SENTENCE_SPLITTER = "/private/home/pkoehn/mosesdecoder/scripts/ems/support/split-sentences.perl -n -q 2>/dev/null";
my $DEDUPE = "/private/home/pkoehn/project/preprocess/build/bin/dedupe 2>/dev/null";
my $dir = "/private/home/pkoehn/data/paracrawl/data";

my ($DIR,$SUBDIR);
die("get-all-mono.perl [-dir DOMAIN_DIR] [-subdir SUBDIR]")
  unless &GetOptions('dir=s' => \$DIR,
                     'subdir=s' => \$SUBDIR);
if (defined($DIR)) {
  &process($DIR);
  exit(0);
}

if (defined($SUBDIR)) {
  &process_subdir($SUBDIR);
  exit(0);
}

# loop over all data
opendir(my $top_ls,$dir);
while(my $subdir = readdir($top_ls)) {
  next if $subdir !~ /^[0-9a-f]{2}$/;
  print "### $subdir ### ".`date`;
  &process_subdir($subdir);
}
close($top_ls);
print "### DONE ###\n";

sub process_subdir {
  my ($subdir) = @_;
  opendir(my $sub_ls,"$dir/$subdir");
  while(my $domain_dir = readdir($sub_ls)) {
    next unless $domain_dir =~ /^(.+)\.20\d\d-\d\d-\d\d$/;
    my $ddir = "$dir/$subdir/$domain_dir";
    &process($ddir);
  }
  closedir($sub_ls);
}

sub process {
  my ($ddir) = @_;

  # already processed?
  return if -e "$ddir/v2.mono.xz";
  return if -e "$ddir/v2.mono.dup";
  return unless -e "$ddir/v2.lett.xz";

  open(LETT,"xzcat $ddir/v2.lett.xz | cut -f 1,6 |");
  while(<LETT>) {
    my ($language,$text) = split(/\t/);
    open(OUT,"| $SENTENCE_SPLITTER -l $language | $DEDUPE >> $ddir/v2.mono.dup");
    my @LINE = split(/\n/,decode_base64($text));
    foreach (@LINE) {
      print OUT $_."\n\n";
    }
    close(OUT);
  }
  close(LETT);
  `cat $ddir/v2.mono.dup | $DEDUPE | xz -T0 - > $ddir/v2.mono.xz`;
  `rm  $ddir/v2.mono.dup`;
}
