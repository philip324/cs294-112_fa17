#!/bin/bash
set -eux

python train_pg.py InvertedPendulum-v1 -n 100 -b 1000 -e 5 -rtg -lr 8e-3 -s 64 --exp_name lb_rtg_na

python train_pg.py InvertedPendulum-v1 -n 100 -b 1000 -e 5 -rtg -bl -lr 8e-3 -s 64 --exp_name lb_rtg_na_bl

