#!/usr/bin/perl -w

use strict;
use Getopt::Long "GetOptions";

my $FASTTEXT = "/private/home/pkoehn/project/fastText-0.9.2/fasttext";
my $MODEL = "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/model.7.bin";
my $THRESHOLD = 0.9;
my $dir = "/private/home/pkoehn/data/paracrawl/data";

my ($DIR,$SUBDIR);
die("language-id-mono-with-fasttext.perl [-dir DOMAIN_DIR] [-subdir SUBDIR]")
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

sub process_subdir {
  my ($subdir) = @_;
  print STDERR "$dir/$subdir\n";
  opendir(my $sub_ls,"$dir/$subdir");
  while(my $domain_dir = readdir($sub_ls)) {
    next unless $domain_dir =~ /^(.+)\.20\d\d-\d\d-\d\d$/;
    my $ddir = "$dir/$subdir/$domain_dir";
    &process($ddir);
  }
  closedir($sub_ls);
}
close($top_ls);
print "### DONE ###\n";

sub process {
  my ($dir) = @_;

  return unless -e "$dir/v2.mono.xz";
  my @STAT = stat "$dir/v2.mono.xz";
  return if time() - $STAT[9] < 300;
  my @ALREADY = `ls $dir/v2.mono.???.xz 2>/dev/null`;
  return if scalar(@ALREADY);

  my %FH;
  open(TEXT,"xzcat $dir/v2.mono.xz|");
  open(LID,"xzcat $dir/v2.mono.xz | $FASTTEXT predict-prob $MODEL - 1 0.0 |");
  while(my $text = <TEXT>) {
    my $lang_score = <LID>;
    $lang_score =~ /__label__(...) ([\d\.]+)/ || die("bad LID output: $lang_score");
    my ($lang,$score) = ($1,$2);
    next if $score < $THRESHOLD;
    if (!defined($FH{$lang})) {
      #open($FH{$lang},"> $dir/v2.mono.$lang");
      open($FH{$lang},"| xz -T0 > $dir/v2.mono.$lang.xz");
    }
    my $fh = $FH{$lang};
    print $fh "$score\t$text";
  }
  foreach my $fh (values %FH) {
    close($fh);
  }
  close(LID);
  close(TEXT); 
}
