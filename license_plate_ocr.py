import sys
import cv2
import numpy as np
import traceback
import darknet.python.darknet as dn

from os.path import splitext, basename
from glob import glob
from darknet.python.darknet import detect, detect_2_0
from src.label import dknet_label_conversion
from src.utils import nms

def ocr(img, ocr_net, ocr_meta, ocr_threshold):
	print('OCR detecting...')
	detected, (width, height) = detect_2_0(ocr_net, ocr_meta, img, thresh=ocr_threshold, nms=None)
	print('OCR detected')
	if len(detected):
		L = dknet_label_conversion(detected, width, height)
		L = nms(L,.45)

		L.sort(key=lambda x: x.tl()[0])
		lp_str = ''.join([chr(l.cl()) for l in L])
		print('\t\tLP: %s' % lp_str)
		return lp_str
	else:
		print('No characters found')
		return None




if __name__ == '__main__':
	try:
		input_dir = sys.argv[1]
		output_dir = input_dir

		ocr_threshold = .4

		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)

		imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

		print('Performing OCR...')

		for i,img_path in enumerate(imgs_paths):
			print('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])

			detected, (width, height) = detect(ocr_net, ocr_meta, img_path, thresh=ocr_threshold, nms=None)

			if len(detected):
				L = dknet_label_conversion(detected, width, height)
				L = nms(L,.45)

				L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])

				with open('%s/%s_str.txt' % (output_dir, bname),'w') as f:
					f.write(lp_str + '\n')

				print('\t\tLP: %s' % lp_str)
			else:
				print('No characters found')
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
