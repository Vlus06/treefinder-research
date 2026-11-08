#!/bin/bash
# build the background-only statistics bank once (needs the label band), then run the usual air worker on the given task file
D=/datastore/phuocnth/treefinder
source $D/env/bin/activate
cd $D/code_air12
OUT=datasets/tile_band_stats_bg.csv
[ -f $OUT ] || python research/tile_band_stats.py --info datasets/tile_info224_v3_aux.csv --tiles datasets/tiles224_v3 --out $OUT --procs 4 --bg_only
bash $D/air_worker12.sh $1
