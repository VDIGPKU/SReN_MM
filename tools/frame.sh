#!/usr/bin/env bash
python gen_result_file.py --name $1 --list /home/hezheqi/data/frame/test/img_list.txt --type frame --poly --iter $3 --dev $2 --imdb test
python eval_frame.py --name $1_poly
