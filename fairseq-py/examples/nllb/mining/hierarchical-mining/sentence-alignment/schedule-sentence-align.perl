#!/usr/bin/perl -w

use strict;

use Getopt::Long "GetOptions";
use FindBin qw($RealBin);
my $ROOTDIR = $RealBin;

my ($VERSION_IN,$VERSION_OUT,$DIR_IN,$DIR_OUT,$DOMAIN,$LANGUAGE,$LOCAL,$DEBUG,$METHOD);
die("sentence-align-hunalign.perl -dir-in DIR -version-in VERSION -dir-out DIR -version-out VERSION [-dir DOMAIN_DIR] [-language ONLY_LANGUAGE]")
  unless &GetOptions('dir-in=s' => \$DIR_IN,
	             'dir-out=s' => \$DIR_OUT,
	             'version-in=s' => \$VERSION_IN,
	             'version-out=s' => \$VERSION_OUT,
		     'domain=s' => \$DOMAIN,
		     'method=s' => \$METHOD,
                     'debug' => \$DEBUG,
                     'language=s' => \$LANGUAGE)
	 && defined($VERSION_IN) && defined($VERSION_OUT)
	 && defined($DIR_IN) && defined($DIR_OUT)
	 && defined($METHOD);

# execution on single domain (specify full path to directory and optionally language)
	     
if (defined($DOMAIN)) {
  if (defined($LANGUAGE)) {
    &process($DOMAIN,$LANGUAGE);
  }
  else {
    &process_all_languages($DOMAIN);
  }
}

# loop over all data

opendir(my $top_ls,$DIR_IN) || die("cannot open $DIR_IN as a directory");
while(my $subdir = readdir($top_ls)) {
  next if $subdir !~ /^[0-9a-f]{2}$/;
  print "### $subdir ### ".`date`;
  opendir(my $sub_ls,"$DIR_IN/$subdir");
  while(my $domain_dir = readdir($sub_ls)) {
    next if $domain_dir =~ /^\./;
    &process_all_languages($subdir,$domain_dir);
  }
  closedir($sub_ls);
}
close($top_ls);
print "### DONE ###\n";

# LOOP OVER FOREIGN LANGUAGES

sub process_all_languages {
  my ($subdir,$domain_dir) = @_;
  my $dir = "$DIR_IN/$subdir/$domain_dir";
  
  # prep had to be done first
  return unless -e "$dir/langstat-digest";

  # must have at least one foreign language (and English)
  my @LANG = `$ROOTDIR/util/get-foreign-languages.perl $dir`;
  return unless scalar(@LANG) >= 1;
  chop(@LANG);

  # process each language
  foreach my $language (@LANG) {
    next if $METHOD eq "bleualign" && ! -e "/private/home/pkoehn/project/paracrawl/document-alignment/models/fast-$language-en/moses.ini";
    if (!defined($LANGUAGE)) {
      &process($subdir,$domain_dir,$language);
    }
    elsif ($LANGUAGE eq $language) {
      &process($subdir,$domain_dir,$language);
    }
  }
}

# MAIN PROCESSING FUNCTION

sub process {
  my ($subdir,$domain_dir,$language) = @_;

  # does input docs exist?
  my $docs = "$DIR_IN/$subdir/$domain_dir/$VERSION_IN.en-$language.docs";
  return unless -e "$docs.xz";

  # does output sent exist or is currently processing?
  my $sent = "$DIR_OUT/$subdir/$domain_dir/$VERSION_OUT.en-$language.sent";
  return if -e "$sent.xz";
  return if -e "$sent.processing";
  return if -e "$sent.scheduled";
  print STDERR "mkdir -p $DIR_OUT/$subdir/$domain_dir\n";
  `mkdir -p $DIR_OUT/$subdir/$domain_dir`;
  `touch $sent.scheduled`;

  # RUN SENTENCE ALIGNMENT
  my $gpu = $METHOD eq "vecalign" ? "--gpus=1" : "--gpus=0";
  `sbatch --job-name sa-$language-$subdir-$domain_dir $gpu --cpus-per-task 1 --partition=nllb --time 24:00:00 --wrap "$ROOTDIR/sentence-align-$METHOD.bash $docs.xz $sent.xz $language"`;
}
