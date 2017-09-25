#!/bin/bash
set -eux
for e in Ant-v1 HalfCheetah-v1 Hopper-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
	for n in 1 5 10 15 20 25 30 35 40 45 50
	do
    	python run_expert.py experts/$e.pkl $e --num_rollouts $n
    done
done