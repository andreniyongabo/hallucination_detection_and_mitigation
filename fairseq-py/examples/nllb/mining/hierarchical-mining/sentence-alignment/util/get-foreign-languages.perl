#!/usr/bin/perl -w

use strict;

# languages to be processed
my %CLD2LANGUAGE = ("ENGLISH" => "en", 
                    "FRENCH" => "fr", 
                    "GERMAN" => "de", 
                    "POLISH" => "pl", 
                    "PORTUGUESE" => "pt", 
                    "DUTCH" => "nl", 
                    "CZECH" => "cs", 
                    "RUSSIAN" => "ru", 
                    "ESTONIAN" => "et", 
                    "SWAHILI" => "sw",
                    "TAGALOG" => "tl",
                    "SOMALI" => "so",
                    "FINNISH" => "fi",
                    "ROMANIAN" => "ro",
                    "LATVIAN" => "lv",
                    "DANISH" => "da",
                    "SWEDISH" => "sv",
                    "GREEK" => "el",
                    "PERSIAN" => "fa",
                    "PASHTO" => "ps",
                    "HAUSA" => "ha",
                    "NEPALI" => "ne",
                    "KHMER" => "km",
                    "VIETNAMESE" => "vi",
                    "SINHALESE" => "si",
                    "BURMESE" => "my",
                    "HUNGARIAN" => "hu",
                    "CROATIAN" => "hr",
                    "SLOVAK" => "sk",
                    "BULGARIAN" => "bg",
                    "SLOVENIAN" => "sl",
                    "LITHUANIAN" => "lt",
                    "IRISH" => "ga",
                    "MACEDONIAN" => "mk",
                    "UKRAINIAN" => "uk",
                    "ARABIC" => "ar",
                    "Chinese" => "zh",
                    "ZULU" => "zu",
                    "KAZAKH" => "kk",
                    "GEORGIAN" => "ka",
                    "Korean" => "ko",
                    "ARABIC" => "ar",
                    "MALTESE" => "mt",
                    "ICELANDIC" => "is",
                    "NORWEGIAN" => "no",
                    "NORWEGIAN_N" => "nn",
                    "ITALIAN" => "it", 
                    "SPANISH" => "es",
                    "AMHARIC" => "am",
                    "OROMO" => "om",
                    "NYANJA" => "ny",
                    "WOLOF" => "wo",
                    "LINGALA" => "ln",
                    "IGBO" => "ig",
                    "SHONA" => "sn",
                    "XHOSA" => "xh",
                    "YORUBA" => "yo",
                    "ZULU" => "zu");

# execution on single domain 
die unless scalar(@ARGV) == 1;
my $dir = $ARGV[0];
die unless -e $dir;
die unless -e "$dir/langstat-digest";

# get information about which languages are in domain
my %DOC;
open(DIGEST,"$dir/langstat-digest");
while(<DIGEST>) {
  my ($cld,$doc_count,$bytes) = split(/\s/);
  $DOC{$CLD2LANGUAGE{$cld}} = $doc_count if defined($CLD2LANGUAGE{$cld});
}
close(DIGEST);

exit unless defined($DOC{"en"}); # must have English

foreach my $language (keys %DOC) {
  print $language."\n" unless $language eq "en";
}
