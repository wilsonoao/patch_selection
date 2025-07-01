#!/bin/bash

for file in /n/scratch/users/f/fas994/wilson/RL/ppo_chief/PAMIL/five_fold/*; do
	echo ${file}
	name_no_ext="${file##*/}"
	name_no_ext="${name_no_ext%.*}"
	python test.py \
               --csv "$file" \
	       --test_dir "/n/scratch/users/f/fas994/wilson/RL/ppo_chief/PAMIL/result_q10_turn3_no_penalty_five_fold/$name_no_ext" \
	       --test_total_T 3 \
	       --action_size 10
done
