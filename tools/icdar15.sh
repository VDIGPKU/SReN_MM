#!/usr/bin/env bash
python gen_result_file.py --name $1 --list /home/hezheqi/Project/frame_regression/data/icdar/test/img_list.txt --type icdar --poly --iter $3 --dev $2
python /home/hezheqi/data/icdar/cvt_format.py $1_poly
/usr/bin/python /home/hezheqi/data/icdar/eval/script.py -g=/home/hezheqi/data/icdar/eval/gt.zip -s=/home/hezheqi/data/icdar/submit/$1_poly.zip
