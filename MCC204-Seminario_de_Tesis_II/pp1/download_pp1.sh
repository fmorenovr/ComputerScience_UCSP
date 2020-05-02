#!usr/bin/bash

PP_URL=http://pulse.media.mit.edu/static/data/

# Place Pulse 1.0
FILENAME_PP1=consolidated_data_jsonformatted.json
FILE_PP1=placepulse_1.json
RESULT_PATH_PP1=./

if [ ! -e $FILE_PP1 ]; then
  wget -P $RESULT_PATH_PP1 $PP_URL$FILENAME_PP1
  mv $FILENAME_PP1 $FILE_PP1
fi
