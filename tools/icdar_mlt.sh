#!/usr/bin/env bash
python gen_result_file.py --name $1 --list /home/hezheqi/Project/frame_regression/data/icdar/test_mlt/img_list.txt --type icdar --poly --iter $3 --dev $2 --imdb mlt
python /home/hezheqi/data/icdar/cvt_format_mlt.py $1_poly
