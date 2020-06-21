import numpy as np

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class YcbDataset(CocoDataset):
	CLASSES = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
	           '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
	           '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
	           '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
	
	def _parse_ann_info(self, img_info, ann_info):
		"""
		overwrite this method.
		process json file to output the info consistent with coco.
		
		we do this process for each one of image, but, you know, the ann_info may be a list.
		"""
		gt_bboxes = []
		gt_labels = []
		gt_bboxes_ignore = []
		gt_masks_ann = []
		
		for i, ann in enumerate(ann_info):
			# the original RLE encoded by cocoAPI gives bytes type of data, which cannot be dumped into json.
			# in data_converter, i used 'ascii' format to decode bytes to string and stored in json.
			# now, i should first encode RLE back using consistent format.
			ann['segmentation']['counts'] = ann['segmentation']['counts'].encode('ascii')
			
			rmin, rmax, cmin, cmax = ann['bbox']
			# bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
			bbox = [cmin, rmin, cmax, rmax]
			
			# if ann.get('iscrowd', False):
			#   gt_bboxes_ignore.append(bbox)
			# else:
			gt_bboxes.append(bbox)
			gt_labels.append(self.cat2label[ann['category_id']])
			gt_masks_ann.append(ann['segmentation'])
		
		if gt_bboxes:
			gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
			gt_labels = np.array(gt_labels, dtype=np.int64)
		else:
			gt_bboxes = np.zeros((0, 4), dtype=np.float32)
			gt_labels = np.array([], dtype=np.int64)
		
		if gt_bboxes_ignore:
			gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
		else:
			gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
		
		seg_map = img_info['filename'].replace('jpg', 'png')
		
		ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann,
		           seg_map=seg_map)
		
		return ann
