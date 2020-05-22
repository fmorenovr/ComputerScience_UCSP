#!usr/bin/bash

PP_URL=http://pulse.media.mit.edu/static/data/

# Place Pulse 2.0
FILENAME_PP2=pp2_20161010.zip
FILE_PP2=placepulse_2.zip
INSIDE_PP2=placepulse_2.csv
RESULT_PATH_PP2=./

if [ ! -e $FILE_PP2 ]; then
  wget -P $RESULT_PATH_PP2 $PP_URL$FILENAME_PP2
  #mv $FILENAME_PP2 $FILE_PP2
fi

unzip $FILENAME_PP2

rm -rf __MACOSX
rm readme
mv votes.csv $INSIDE_PP2

rm $FILENAME_PP2
zip $FILE_PP2 $INSIDE_PP2
rm $INSIDE_PP2
