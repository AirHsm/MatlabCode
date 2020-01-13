import PySimpleGUI as sg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF
import os
from PIL import Image
from imutils import build_montages


## Global
global window
global event
global values




# set theme
sg.theme('BluePurple')

# callback functions

def CircleDetection():
	global circles
	global img
	global fimg
	global RR
	#imgdir, fimgdir, minr, maxr
	img = cv2.imread(values['_IMGDIR_'],0)
	fimg = cv2.imread(values['_FIMGDIR_'],0)

	minr = int(values['_MINRADIUS_'])
	maxr = int(values['_MAXRADIUS_'])

	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 100, minRadius=minr, maxRadius=maxr)
	circles = np.uint16(np.around(circles))

	Rs = circles[0,:,2]
	RR = Rs.min()
	window.Element('_RR_').update(str(RR))
	window.Element('_CIRCLESNUM').update(str(circles.shape[1]))
	window.Element('_RMIN_').update(str(Rs.min()))
	window.Element('_RMAX_').update(str(Rs.max()))


def DrawCircles():
	# gray to BGR
	cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cfimg = cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR)
	# draw circle
	for i in circles[0,:]:
		cv2.circle(cimg, (i[0],i[1]), i[2], (255,0,0),2)
		cv2.circle(cimg, (i[0],i[1]), 2, (255,0,255),5)
		cv2.circle(cfimg, (i[0],i[1]), i[2], (255,0,0),2)

	plt.figure()
	plt.subplot(121),plt.title('Bright field image with circles'), plt.imshow(cimg)
	plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.title('Fluorescent image with detected circles'), plt.imshow(cfimg)
	plt.xticks([]), plt.yticks([])
	plt.show(block=False)


def analysis():

	global DropDensity
	global DropSTD
	global Drops
	global dropsnum

	DropDensity = list()
	DropSTD = list()
	Drops = list()

	m, n = img.shape
	x = np.linspace(0, n-1, n)
	y = np.linspace(0, m-1, m)
	X,Y = np.meshgrid(x, y)

	for i in range(circles.shape[1]):
		centx = circles[0, i, 0]
		centy = circles[0, i, 1]
		r = circles[0, i, 2]
		if 1+RR<centx<n-RR and 1+RR<centy<m-RR:
			Rv = np.sqrt((X-centx)**2 + (Y-centy)**2)
			drop = fimg[centy-RR:centy+RR,centx-RR:centx+RR]
			Drops.append(drop)
			DropSTD.append(np.std(drop))
			DropDensity.append(np.sum(fimg*(Rv<=RR))/RR**2)

	window.Element('_DROPNUM_').update(str(len(Drops)))

	THR = int(np.fix(np.max(DropDensity)))
	window.Element('_THR_').update(str(THR))

	'''
	plt.figure()
	plt.subplot(131), plt.imshow(Drops[1]), plt.xticks([]), plt.yticks([])
	plt.subplot(132)
	plt.hist(DropDensity),plt.title('Fluorescence density')
	plt.subplot(133)
	plt.hist(DropSTD), plt.title('Fluorescence STD')
	plt.show(block=False)
	'''

def CellNumCount():
	print('Starting count...')

	global Drops0
	global Drops1
	global Drops2

	Drops0 = list()
	Drops1 = list()
	Drops2 = list()

	THR = int(values['_THR_'])

	for drop in Drops:
		ret, thresh = cv2.threshold(drop, THR, 255, 0)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contourNum = len(contours)
		if contourNum == 0:
			Drops0.append(drop)
		if contourNum == 1:
			rect = cv2.minAreaRect(contours[0])
			WH = rect[1]
			if np.max(WH)/np.min(WH) > 1.3:
				Drops2.append(drop)
			else:
				Drops1.append(drop)
		if contourNum > 1:
			Drops2.append(drop)

	window.Element('_DROPNUM0_').update(str(len(Drops0)))
	window.Element('_DROPNUM1_').update(str(len(Drops1)))
	window.Element('_DROPNUM2_').update(str(len(Drops2)))
	return


def ExportData():
	filepath =  values['_EXPORT_']
	WriteTIFF(os.path.join(filepath,'Drops.tif'), Drops)	# Export images of each drops into one .tif file

	return


def ReadTIFF(filename):
	tif = TIFF.open(filename, mode='r')
	Imgs = list()
	for img in list(tif.iter_images()):
		Imgs.append(img)
	return

def WriteTIFF(filename, Imgs):
	tif = TIFF.open(filename, mode='w')
	ImgsNum = len(Imgs)
	for img in Imgs:
		tif.write_image(img, compression=None)
	tif.close()
	return


# layout

layout = [
			[sg.Text('数据导入')],
			[
				sg.Text('明场图像位置',size=(10,1)),
				sg.Input('/Users/hushiming/workspace/personal/projects/SomeCodes/液滴检测/data/10x-1.tif', key='_IMGDIR_', size=(50,1)), sg.FileBrowse('打开', target='_IMGDIR_', size=(5,1)),
				sg.Button('显示明场图像',size=(10,1))],
			[
				sg.Text('荧光图像位置',size=(10,1)),
				sg.Input('/Users/hushiming/workspace/personal/projects/SomeCodes/液滴检测/data/10x-2.tif', key='_FIMGDIR_', size=(50,1)),
				sg.FileBrowse('打开', target='_FIMGDIR_', size=(5,1)),
				sg.Button('显示荧光图像',size=(10,1))],
			[
				sg.Text('液滴识别')],

			[
				sg.Text('液滴半径范围 (pixel):',size=(10,2)),
				sg.Text('Min:', size=(3,1)),
				sg.Input('58', key='_MINRADIUS_', size=(3,1)),
				sg.Text('Max:', size=(5,1)),
				sg.Input('62', key='_MAXRADIUS_', size=(5,1)),
				sg.Button('开始识别')],
			[
				sg.Text('识别结果'),
				sg.Text('Rmin:'),
				sg.Text('**', key='_RMIN_'),
				sg.Text('Rmax:'),
				sg.Text('**', key='_RMAX_'),
				sg.Text('识别到圆形总数：'),
				sg.Text('***', key='_CIRCLESNUM')],
			[
				sg.Text('分析')],
			[
				sg.Text('输入有效半径（默认检测最小半径）：', size=(10,3)),
				sg.Input(key='_RR_',size=(5,1)),
				sg.Button('开始分析',size=(10,1)),
				sg.Text('阈值(Max=255):'),
				sg.Input(key='_THR_',size=(5,1)),
				sg.Button('区分液滴')],
			[
				sg.Text('完整液滴总数：',size=(10,1)),
				sg.Text('***', size=(5,1), key='_DROPNUM_'),
				sg.Text('无细胞液滴个数:'),
				sg.Text('***', size=(5,1), key='_DROPNUM0_'),
				sg.Text('单细胞液滴个数:'),
				sg.Text('***', size=(5,1), key='_DROPNUM1_'),
				sg.Text('多细胞液滴个数:'),
				sg.Text('***', size=(5,1), key='_DROPNUM2_')],
			[
				sg.Button('显示液滴',size=(10,1)),
				sg.Button('显示无细胞液滴',size=(10,1)),
				sg.Button('显示单细胞液滴',size=(10,1)),
				sg.Button('显示多细胞液滴',size=(10,1))],
			[
				sg.Button('总荧光分布直方图', size=(20,1)),
				sg.Button('无细胞荧光分布直方图', size=(20,1)),
				sg.Button('单细胞荧光分布直方图', size=(20,1)),
				sg.Button('多细胞荧光分布直方图', size=(20,1)),],
			[
				sg.Text('数据导出路径', size=(10,1)),
				sg.Input('/Users/hushiming/workspace/personal/projects/SomeCodes/液滴检测', key='_EXPORT_', size=(50,1)),
				sg.FolderBrowse('打开',target='_EXPORT_',size=(5,1)),
				sg.Button('Export', size=(5,1))],
			[
				sg.Text(key='_EXPORTSTATUS_')],
			[
				sg.Multiline(key='_OUTPUT_', size=(60,2)), sg.Exit()],

]

window = sg.Window('液滴检测', layout)

while True:
	event, values = window.read()
	if event is None or event == "Exit":
		cv2.destroyAllWindows()
		break


	if event == '显示明场图像':
		try:
			IMG = cv2.imread(values['_IMGDIR_'],0)
			plt.figure()
			plt.imshow(IMG,'gray'), plt.title('Bright field image')
			plt.show(block=False)
		except Exception as e:
			window.Element('_OUTPUT_').update('无法打开明场图像')

	if event == '显示荧光图像':
		try:
			FIMG = cv2.imread(values['_FIMGDIR_'],0)
			plt.figure()
			plt.imshow(FIMG, 'gray'), plt.title('Fluorescent image')
			plt.show(block=False)
		except Exception as e:
			window.Element('_OUTPUT_').update('无法打开荧光图像')

	if event == '开始识别':
		try:
			CircleDetection()
			try:
				DrawCircles()
			except Exception as e:
				window.Element('_OUTPUT_').update('Cannot draw the circles.')

		except Exception as e:
			window.Element('_OUTPUT_').update('Cannot find any circles.\nPlease check the input data or radius range of the drops.')
			pass

	#if event == 'Close Data Figure':
	#	cv2.destroyAllWindows()
	if event == '开始分析':
		try:
			analysis()
		except Exception as e:
			window.Element('_OUTPUT_').update('Something is wrong.')
	if event == '区分液滴':
		try:
			CellNumCount()
		except Exception as e:
			window.Element('_OUTPUT_').update('Cannot count cells.')

	if event == 'Export':
		try:
			ExportData()
			window.Element('_EXPORTSTATUS_').update('Data export complete.')
		except Exception as e:
			window.Element('_OUTPUT_').update('Cannot export data successfully. Please check.')

	if event == '显示液滴':
		#montage = build_montages(imgs, (m,n), (x,x))
		pass



window.Close()

#sg.Popup('The GUI returned:', event, values)
