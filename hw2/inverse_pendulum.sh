#!/bin/bash
set -eux

# not bad
# python train_pg.py InvertedPendulum-v1 -n 100 -b 1000 -e 5 -rtg -dna -lr 7e-3 --exp_name sb_rtg_dna

python train_pg.py InvertedPendulum-v1 -n 100 -b 1000 -e 5 -dna -lr 8e-3 -s 64 --exp_name sb_no_rtg_dna

python train_pg.py InvertedPendulum-v1 -n 100 -b 1000 -e 5 -rtg -dna -lr 8e-3 -s 64 --exp_name sb_rtg_dna

python train_pg.py InvertedPendulum-v1 -n 100 -b 1000 -e 5 -rtg -lr 8e-3 -s 64 --exp_name sb_rtg_na

python train_pg.py InvertedPendulum-v1 -n 100 -b 5000 -e 5 -dna -lr 8e-3 -s 64 --exp_name lb_no_rtg_dna

python train_pg.py InvertedPendulum-v1 -n 100 -b 5000 -e 5 -rtg -dna -lr 8e-3 -s 64 --exp_name lb_rtg_dna

python train_pg.py InvertedPendulum-v1 -n 100 -b 5000 -e 5 -rtg -lr 8e-3 -s 64 --exp_name lb_rtg_na

