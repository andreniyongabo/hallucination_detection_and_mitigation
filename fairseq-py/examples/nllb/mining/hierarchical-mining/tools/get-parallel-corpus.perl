#!/usr/bin/perl

use strict;
use Getopt::Long "GetOptions";

my ($LANGUAGE);
my $VERSION=2;

die
  unless &GetOptions('language=s' => \$LANGUAGE,
	             'version=s' => \$VERSION);
die unless defined($LANGUAGE);

my $dir = "/private/home/pkoehn/data/paracrawl/data";

opendir(my $top_ls,$dir);
while(my $subdir = readdir($top_ls)) {
  next if $subdir !~ /^[0-9a-f]{2}$/;
  print STDERR "### $subdir ### ".`date`;
  opendir(my $sub_ls,"$dir/$subdir");
  while(my $domain_dir = readdir($sub_ls)) {
    next if $domain_dir =~ /^\./;
    my $ddir = "$dir/$subdir/$domain_dir";
    next unless -e "$ddir/v$VERSION.en-$LANGUAGE.sent.xz";
    open(EF,"xzcat $ddir/v$VERSION.en-$LANGUAGE.sent.xz |");
    while(my $ef = <EF>) {
      chop($ef);
      my @EF = split(/\t/,$ef);
      if (scalar(@EF) != 5) {
        print STDERR "ERROR: buggy line in $ddir/v$VERSION.en-$LANGUAGE.sent.xz, only ".scalar(@EF)." items, should be 4\n";
        last;
      }
      next if $EF[2] eq "" || $EF[3] eq "";
      print $domain_dir."\t".$ef."\n";
    }
    close(Z);
    close(EF);
  }
  closedir($sub_ls);
}
close($top_ls);
