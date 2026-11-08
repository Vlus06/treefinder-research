#!/bin/bash
# Per-tile band statistics (statistics bank table) for every tile; labels are not read.
D=/datastore/phuocnth/treefinder
cd $D/code_air
[ -f $D/datasets/tile_band_stats.csv ] || $D/env/bin/python research/tile_band_stats.py --info datasets/tile_info224_v3_aux.csv \
    --tiles datasets/tiles224_v3 --out datasets/tile_band_stats.csv --procs 4
wc -l $D/datasets/tile_band_stats.csv
