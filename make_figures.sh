#!/bin/sh
cd figures
for fn in fig_*.tex; do
    lualatex $fn
done
cd ..
