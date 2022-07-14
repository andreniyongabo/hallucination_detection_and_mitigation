#!/usr/bin/env perl
use strict;
use warnings;
use utf8;
binmode STDIN, ':utf8';
binmode STDOUT, ':utf8';

my $soft_hyphen = chr(0xAD);
while(<>){
    s/(^|[ ({[])("|,,|''|``)/$1„/g;
    s/("|''|‟)($|[ ,.?!:;)}\]])/“$2/g;
    s/ --? / – /g;
    s/$soft_hyphen//g;
    print;
}
