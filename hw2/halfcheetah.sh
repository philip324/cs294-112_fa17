#!/bin/bash
set -eux

# avg 80
# python train_pg.py HalfCheetah-v1 -ep 150 -n 100 -b 50000 -rtg --discount 0.95 -bl -lr 0.02 -s 64 -l 2

# avg 90
# python train_pg.py HalfCheetah-v1 -ep 150 -n 100 -b 50000 -rtg --discount 0.90 -bl -lr 0.025 -s 64 -l 2

# avg 110 cheetah2
# python train_pg.py HalfCheetah-v1 -ep 150 -n 100 -b 50000 -rtg --discount 0.95 -bl -lr 0.025 -s 64 -l 2

# avg 120 cheetah1
# python train_pg.py HalfCheetah-v1 -ep 150 -n 100 -b 50000 -rtg --discount 0.95 -bl -lr 0.03 -s 64 -l 2


python train_pg.py HalfCheetah-v1 -ep 150 -n 100 -b 50000 -rtg --discount 0.95 -bl -lr 0.028 -s 64 -l 3
