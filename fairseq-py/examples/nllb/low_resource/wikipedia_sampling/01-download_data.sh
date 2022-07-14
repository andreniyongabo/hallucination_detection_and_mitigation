#!/bin/bash

mkdir lists

# Download list of articles considered vital for all Wikipedias
BASE=https://meta.wikimedia.org/wiki/List_of_articles_every_Wikipedia_should_have/Expanded
PAGES="People History Geography Arts Philosophy_and_religion Anthropology,_psychology_and_everyday_life Society_and_social_sciences Biology_and_health_sciences Physical_sciences Technology Mathematics"
for page in ${PAGES}; do
  echo ${BASE}/${page}
  out=raw/Global:${page}.html
  if [ ! -f $out ]; then
    wget -O $out ${BASE}/${page}
    sleep 1s
  fi
done

# Download list of articles considered vital for English Wikipedia.
# NB: This is commented out as the list above is a better choice – this one here is
# specific to English Wikipedia and has a worse coverage of multicultural topics.
# BASE=https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4
# PAGES="People History Geography Arts Philosophy_and_religion Everyday_life Society_and_social_sciences Biology_and_health_sciences Physical_sciences Technology Mathematics"
# for page in ${PAGES}; do
#   echo ${BASE}/${page}
#   out=raw/English:${page}.html
#   if [ ! -f $out ]; then
#     wget -O $out ${BASE}/${page}
#     sleep 1s
#   fi
# done

# Get Wikidata dump
if [ ! -f raw/wikidata.json.gz ]; then
  wget -O raw/wikidata.json.gz https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz
fi

# Get Wikipedia dump
if [ ! -f raw/wikipedia_dump.xml.bz2 ]; then
  wget -O raw/wikipedia_dump.xml.bz2 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
fi
if [ ! -f raw/wikipedia_dump.xml ]; then
  bunzip2 raw/wikipedia_dump.xml.bz2
fi
