import cv2
import os.path as osp
import pickle
import sys
import os
import argparse


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Generate txt result file')
  parser.add_argument('--name', dest='out_name',
                      help='name of output', default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16 or res101', default='res101', type=str)
  parser.add_argument('--list', dest='img_list_dir',
                      help='image list', default='/data/hezheqi/frame/test/img_list.txt', type=str)
  parser.add_argument('--type', dest='type',
                      help='frame or icdar', default='frame', type=str)
  parser.add_argument('--iter', dest='iter',
                      help='iter', default=70000, type=int)
  parser.add_argument('--pkl', dest='pkl_dir',
                      help='pkl dir',
                      default='',
                      type=str)
  parser.add_argument('--imdb', dest='imdb',
                      help='imdb',
                      default='test', type=str)
  parser.add_argument('--poly', dest='poly_mode', help='poly mode',
                      action='store_true')
  parser.add_argument('--dev', dest='dev', help='dev name',
                      default="default", type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def gen_result_file(out_name, pkl_dir, poly=False, dataset='frame'):
  results = pickle.load(open(osp.join(pkl_dir, 'detections.pkl'), 'rb'))
  out_dir = '/home/hezheqi/data/{}/result/{}'.format(dataset, out_name)
  print(out_dir)
  if poly:
    out_dir += '_poly'
  if not osp.isdir(out_dir):
    os.makedirs(out_dir)
  # return
  boxes = results[1]
  with open(args.img_list_dir) as fin:
    for i, name in enumerate(fin):
      name = name.strip()
      res = []
      fout = open(osp.join(out_dir, name + '.txt'), 'w')
      for box in boxes[i]:
        # if float(box[-1]) < 0.85:
        #   continue
        if not poly:
          x1, y1, x2, y2 = [str(int(b)) for b in box[:4]]
          res.append('4 ' + x1 + ' ' + y1 + ' ' + x2 + ' ' + y1 + ' ' +
                     x2 + ' ' + y2 + ' ' + x1 + ' ' + y2 + ' ' + str(box[-1])+ '\n')
        else:
          box_str = ' '.join([str(int(b)) for b in box[:-1]])
          box_str += ' ' + str(box[-1])
          res.append('4 ' + box_str + '\n')
      fout.write(str(len(res)) + '\n')
      for r in res:
        fout.write(r)
      fout.close()


if __name__ == '__main__':
  args = parse_args()
  # gen_result_file_with_label(sys.argv[1],sys.argv[2], iter_num)
  # gen_traffic_file(sys.argv[1],sys.argv[2], iter_num)
  imdb = args.imdb
  if imdb == 'test':
    imdb = 'test'
  elif 'text' in args.type:
    imdb = imdb
  elif '17' not in imdb:
      imdb = 'test_' + imdb
  else:
      imdb = imdb
  print(imdb)
  if len(args.pkl_dir) == 0:
      args.pkl_dir = osp.join(osp.dirname(__file__), '..')

  dev_tpye = args.dev
  # pkl_dir = osp.join(args.pkl_dir, 'output/{}/{}_{}/faster_rcnn_epoch_{}'
  #                    .format(args.net, args.type, imdb, args.iter))
  pkl_dir = osp.join(args.pkl_dir, 'output/{}/{}_{}/{}/faster_rcnn_epoch_{}'
                     .format(args.net, args.type, imdb, dev_tpye, args.iter))
 
  gen_result_file(args.out_name, pkl_dir=pkl_dir, poly=args.poly_mode, dataset=args.type)
  # gen_vechicle_result('test')
