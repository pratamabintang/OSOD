from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import scipy.sparse
from PIL import Image

from datasets.imdb import imdb
from model.utils.config import cfg


class broiler(imdb):
  """Single-class (broiler) YOLO-format dataset, RGB only.

  Layout: <DATA_DIR>/broiler/<split>/{rgb,label}/<name>.{png,txt}
  Label line: cls cx cy w h  (normalized). Class 0 is mapped to index 1.
  """

  def __init__(self, image_set, data_path=None):
    imdb.__init__(self, 'broiler_' + image_set)
    self._image_set = image_set
    self._data_path = data_path if data_path is not None \
      else os.path.join(cfg.ROOT_DIR, 'datasets', 'broiler')
    self._classes = ('__background__', 'broiler')
    self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
    self._image_ext = '.png'
    self._image_index = self._load_image_set_index()
    self._roidb_handler = self.gt_roidb
    self.config = {'cleanup': True, 'use_salt': True}

    self.cat_data = {}
    for i in range(self.num_classes):
      self.cat_data[i] = []

  def _split_dir(self):
    return os.path.join(self._data_path, self._image_set)

  def _load_image_set_index(self):
    rgb_dir = os.path.join(self._split_dir(), 'rgb')
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(rgb_dir)
                   if f.endswith(self._image_ext))
    return names

  def image_path_at(self, i):
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    return i

  def image_path_from_index(self, index):
    path = os.path.join(self._split_dir(), 'rgb', index + self._image_ext)
    assert os.path.exists(path), 'Path does not exist: {}'.format(path)
    return path

  def gt_roidb(self):
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        [roidb, self.cat_data] = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_annotation(index) for index in self._image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump([gt_roidb, self.cat_data], fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_annotation(self, index):
    im_path = self.image_path_from_index(index)
    width, height = Image.open(im_path).size

    label_path = os.path.join(self._split_dir(), 'label', index + '.txt')
    rows = []
    if os.path.exists(label_path):
      with open(label_path) as f:
        for line in f:
          parts = line.split()
          if len(parts) == 5:
            rows.append([float(x) for x in parts])

    num_objs = len(rows)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    for ix, (_, cx, cy, w, h) in enumerate(rows):
      x1 = max(0, (cx - w / 2.) * width)
      y1 = max(0, (cy - h / 2.) * height)
      x2 = min(width - 1, (cx + w / 2.) * width - 1)
      y2 = min(height - 1, (cy + h / 2.) * height - 1)
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = 1
      overlaps[ix, 1] = 1.0
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
      self.cat_data[1].append({'boxes': [x1, y1, x2, y2],
                               'image_path': im_path})

    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def evaluate_detections(self, all_boxes, output_dir):
    aps = []
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      ap = self._voc_ap_for_class(all_boxes[cls_ind], cls_ind)
      aps.append(ap)
      print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    return aps

  def _voc_ap_for_class(self, dets_per_image, cls_ind, ovthresh=0.5):
    # GT per image for this class, from roidb (no XML needed).
    gt = {}
    npos = 0
    for im_ind, index in enumerate(self._image_index):
      entry = self.roidb[im_ind]
      mask = entry['gt_classes'] == cls_ind
      bbox = entry['boxes'][mask].astype(np.float64)
      gt[im_ind] = {'bbox': bbox, 'det': np.zeros(len(bbox), dtype=bool)}
      npos += len(bbox)

    # Flatten detections: (im_ind, score, x1, y1, x2, y2)
    rows = []
    for im_ind, dets in enumerate(dets_per_image):
      if len(dets) == 0:
        continue
      for d in dets:
        rows.append([im_ind, d[-1], d[0], d[1], d[2], d[3]])
    if len(rows) == 0 or npos == 0:
      return 0.0
    rows = np.array(rows)
    rows = rows[np.argsort(-rows[:, 1])]

    nd = rows.shape[0]
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
      im_ind = int(rows[d, 0])
      bb = rows[d, 2:6]
      BBGT = gt[im_ind]['bbox']
      ovmax = -np.inf
      if BBGT.size > 0:
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not gt[im_ind]['det'][jmax]:
          tp[d] = 1.
          gt[im_ind]['det'][jmax] = True
        else:
          fp[d] = 1.
      else:
        fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return _voc_ap(rec, prec)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True

  def filter(self, seen=1):
    # single class: index 1 is used for train and test
    self.list = [1]
    self.inverse_list = [1]


def _voc_ap(rec, prec):
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
  i = np.where(mrec[1:] != mrec[:-1])[0]
  return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
