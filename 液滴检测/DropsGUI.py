import PySimpleGUI as sg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF
import os
from PIL import Image
from imutils import build_montages
import xlwt
import pandas as pd
from scipy import misc
import time


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
	global imgo 
	global fimgo 
	global RR
	global FIMG
	#imgdir, fimgdir, minr, maxr

	if values['_ISW_']:
		I = Image.open(values['_WIMGDIR_'])
		I.seek(0)
		temp = np.array(I)
		temp = temp/(2**16)*255
		img = np.array(temp,dtype=np.uint8)
		I.seek(1)
		temp = np.array(I)
		temp = temp/(2**16)*255
		fimg = np.array(temp,dtype=np.uint8)
		FIMG = np.array(I)
		#plt.figure(),plt.subplot(121),plt.imshow(img,'gray'),plt.subplot(122),plt.imshow(fimg,'gray'),plt.show(block=False)
		#pass
	else:
		img = cv2.imread(values['_IMGDIR_'],0)
		fimg = cv2.imread(values['_FIMGDIR_'],0)
		FIMG = cv2.imread(values['_FIMGDIR_'],-1)
	#img = misc.imread(values['_IMGDIR_'])
	#FIMG = misc.imread(values['_FIMGDIR_'])

	# intensity calibration
	if values['_isIC_']:
		img_c = cv2.imread('IC.tif',0)
		fimg_c = cv2.imread('fIC.tif',0)
		imgo = img;
		fimgo = fimg;
		img = IntensityCorrection(img, img_c)
		fimg = IntensityCorrection(fimg, fimg_c)

	#test
	img= cv2.resize(img,(2048,2048),interpolation=cv2.INTER_CUBIC)
	fimg= cv2.resize(fimg,(2048,2048),interpolation=cv2.INTER_CUBIC)
	#FIMG = cv2.resize(FIMG,(2048,2048),interpolation=cv2.INTER_CUBIC)

	minr = int(values['_MINRADIUS_'])
	maxr = int(values['_MAXRADIUS_'])

	

	#test
	img = cv2.equalizeHist(img)
	#ret, img = cv2.threshold(img,110,255,cv2.THRESH_BINARY)

	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, minr*2-2, param1=100, param2=50, minRadius=minr, maxRadius=maxr)
	circles = np.uint16(np.around(circles))

	Rs = circles[0,:,2]
	RR = Rs.min()
	window.Element('_RR_').update(str(RR))
	window.Element('_CIRCLESNUM').update(str(circles.shape[1]))
	window.Element('_RMIN_').update(str(Rs.min()))
	window.Element('_RMAX_').update(str(Rs.max()))


def IntensityCorrection(a, img_c):
	Ic = 1/(img_c/img_c.max())
	return np.array(a * Ic, dtype=np.uint8)


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
	global DropsIndex
	global circles

	DropDensity = list()
	DropSTD = list()
	Drops = list()
	DropsIndex = list()

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
			temp = fimg * (Rv <= RR)
			drop = temp[centy-RR:centy+RR,centx-RR:centx+RR]
			Drops.append(drop)
			DropSTD.append(np.std(drop))
			DropDensity.append(np.sum(fimg*(Rv<=RR))/RR**2)
			DropsIndex.append([centx,centy,r])

	window.Element('_DROPNUM_').update(str(len(Drops)))

	THR = int(np.fix(np.max(DropDensity)))
	window.Element('_THR_').update(str(THR))

	print(DropsIndex[0])
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
	# detect cell number of each drops.
	print('Starting count...')

	global Drops0
	global Drops1
	global Drops2
	global DropCellNum

	global Index0
	global Index1
	global Index2

	Drops0 = list()
	Drops1 = list()
	Drops2 = list()
	DropCellNum = list()

	Index0 = list()
	Index1 = list()
	Index2 = list()

	THR = int(values['_THR_'])
	DropCellNum = np.zeros([len(Drops),1], dtype=np.int)
	for i, drop in enumerate(Drops):

		ret, thresh = cv2.threshold(drop, THR, 255, 0)
		try :
			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			contourNum = len(contours)
			#if contourNum == 0:
			#	Drops0.append(drop)
			#	Index0.append(circles[0,i,:])
			if contourNum == 1:
				rect = cv2.minAreaRect(contours[0])
				WH = rect[1]
				if np.max(WH)/np.min(WH) > 1.3:
					Drops2.append(drop)
					Index2.append(circles[0,i,:])
					DropCellNum[i] = 2
				else:
					Drops1.append(drop)
					Index1.append(circles[0,i,:])
					DropCellNum[i] = 1
			elif contourNum > 1:
				Drops2.append(drop)
				index2.append(circles[0,i,:])
				DropCellNum[i] = 2
			else:
				Drops0.append(drop)
				Index0.append(circles[0,i,:])
		except:
			Drops0.append(drop)
			Index0.append(circles[0,i,:])

	print(len(Drops))
	print(len(DropCellNum))


	print(len(Index0))
	print(np.shape(Index0[0]))

	window.Element('_DROPNUM0_').update(str(len(Drops0)))
	window.Element('_DROPNUM1_').update(str(len(Drops1)))
	window.Element('_DROPNUM2_').update(str(len(Drops2)))
	return


def ReadTIFF(filename):
	# read images from a .tif file
	tif = TIFF.open(filename, mode='r')
	Imgs = list()
	for img in list(tif.iter_images()):
		Imgs.append(img)
	return


def WriteTIFF(filename, Imgs):
	# write image into a .tif file
	tif = TIFF.open(filename, mode='w')
	ImgsNum = len(Imgs)
	for img in Imgs:
		tif.write_image(img, compression=None)
	tif.close()
	return


def MakeMontage(Imgs):
	# make montage view of drop images
	m,n = np.shape(Imgs[0])
	num = int(np.ceil(np.sqrt(len(Imgs))))
	montage = np.zeros((m*num,n*num))
	i = 0
	count = len(Imgs)
	for x in range(num):
		for y in range(num):
			if i >= count:
				break
			montage[y*m:(y+1)*m, x*n:(x+1)*n] = Imgs[i]
			i += 1
	print('Montage generated.')
	print(num)
	return montage


def ShowDrops(CellNum):
	# show images of drops for checking
	#cI = cfimg
	if CellNum == -1:
		Imgs = Drops
		#for i in circles[0,:]:
		#	cv2.circle(cI, (i[0],i[1]), i[2], (255,0,0),2)
	else:
		if CellNum == 0:
			Imgs = Drops0
		#	cIndex = Index0
		if CellNum == 1:
			Imgs = Drops1
		#	cIndex = Index1
		if CellNum > 1:
			Imgs = Drops2
		#	cIndex = Index2
		#for i in cIndex:
		#	cv2.circle(cI, (i[0], i[1]), i[2], (255,0,0),2)
	montage = MakeMontage(Imgs)
	plt.figure()
	#lt.subplot(121), plt.imshow(cI), plt.xticks([]), plt.yticks([])
	#plt.subplot(122), plt.imshow(montage, 'gray'), plt.xticks([]), plt.yticks([])
	Ts = str(CellNum) + 'cells'
	plt.imshow(montage, 'gray'), plt.xticks([]), plt.yticks([]),plt.title(Ts)
	plt.show(block=False)


def ShowHist(CellNum):
	if CellNum == -1:
		Imgs = Drops
	if CellNum == 0:
		Imgs = Drops0
	if CellNum == 1:
		Imgs = Drops1
	if CellNum > 1:
		Imgs = Drops2

	intensity = list()

	for img in Imgs:
		intensity.append(np.sum(img))

	plt.figure()
	plt.hist(intensity)
	plt.show(block=False)


# Excel
def SetStyle(name, height, bold=False):
	style = xlwt.XFStyle()
	font = xlwt.Font()
	font.name = name
	font.bold = bold
	font.color_index = 4
	font.height = height
	style.font = font
	return style


def Export2Excel(savingpath):
	f = xlwt.Workbook()
	sheet1 = f.add_sheet('Drops', cell_overwrite_ok=True)
	row0 = ['No.', 'DropR', 'Location', 'CellNumber', 'MeanIntensity']
	dropsnum = len(Drops)
	for i in range(0, len(row0)):
		sheet1.write(0,i,row0[i],SetStyle('Times New Roman',220,True))
	for i in range(0,dropsnum):
		# No.
		sheet1.write(i+1,0,i+1,SetStyle('Times New Roman',220,True))
		# DropR
		sheet1.write(i+1,1,str(RR),SetStyle('Times New Roman',220,True))
		# Location
		Location = str(DropsIndex[i])
		sheet1.write(i+1,2,Location,SetStyle('Times New Roman',220,True))
		#CellNumber
		sheet1.write(i+1,3,str(DropCellNum[i]),SetStyle('Times New Roman',220,True))
		#MeanIntensity
		MeanIntensity = str(DropDensity[i])
		sheet1.write(i+1,4,MeanIntensity,SetStyle('Times New Roman',220,True))
	f.save(savingpath)
	pass


def Updating(text):
	OutputtingStr = []
	try: 
		OutputtingStr += test + '\n'
		window.Element('_OUTPUT_').update(OutputtingStr)
	except Exception as e:
		pass

# layout

layout = [
			[sg.Text('数据导入')],
			[
				sg.Text('明场图像位置',size=(10,1)),
				sg.Input('/Users/hushiming/OpenSource/SomeCodes/液滴检测/data/10x-1.tif', key='_IMGDIR_', size=(50,1)), sg.FileBrowse('打开', target='_IMGDIR_', size=(5,1)),
				sg.Button('显示明场图像',size=(10,1))],
			[
				sg.Text('荧光图像位置',size=(10,1)),
				sg.Input('/Users/hushiming/OpenSource/SomeCodes/液滴检测/data/10x-2.tif', key='_FIMGDIR_', size=(50,1)),
				sg.FileBrowse('打开', target='_FIMGDIR_', size=(5,1)),
				sg.Button('显示荧光图像',size=(10,1))],
			[
				sg.Checkbox('是否是合并图像位置',default=True, key='_ISW_'),
				sg.Text('合并图像位置', size=(10,1)),
				sg.Input('hello', key='_WIMGDIR_', size=(10,1)),
				sg.FileBrowse('打开', target='_WIMGDIR_', size=(5,1))],
			[
				sg.Checkbox('荧光校正', default=True, key='_isIC_'),
			],	
			[
				sg.Text('液滴识别')],

			[
				sg.Text('液滴半径范围 (pixel):',size=(10,2)),
				sg.Text('Min:', size=(3,1)),
				sg.Input('40', key='_MINRADIUS_', size=(3,1)),
				sg.Text('Max:', size=(5,1)),
				sg.Input('44', key='_MAXRADIUS_', size=(5,1)),
				sg.Button('开始识别'),
				sg.Button('Run by default'),
				sg.Button('显示校正效果', size=(10,1)),
				sg.Button('导出校正后荧光图像', size=(10,1))],
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
				sg.Button('显示液滴',size=(20,1)),
				sg.Button('显示无细胞液滴',size=(20,1)),
				sg.Button('显示单细胞液滴',size=(20,1)),
				sg.Button('显示多细胞液滴',size=(20,1))],
			[
				sg.Button('总荧光分布直方图', size=(20,1)),
				sg.Button('无细胞荧光分布直方图', size=(20,1)),
				sg.Button('单细胞荧光分布直方图', size=(20,1)),
				sg.Button('多细胞荧光分布直方图', size=(20,1)),],
			[
				sg.Text('数据导出路径', size=(10,1)),
				sg.Input('/Users/hushiming/OpenSource/SomeCodes/液滴检测/data', key='_EXPORT_', size=(50,1)),
				sg.FolderBrowse('打开',target='_EXPORT_',size=(5,1)),
				sg.Button('Export', size=(5,1))],
			[
				sg.Text(key='_EXPORTSTATUS_')],
			[
				sg.Multiline(key='_OUTPUT_', size=(60,2)), sg.Exit()],

]

window = sg.Window('液滴检测', layout)
OutputtingStr = ''

while True:
	event, values = window.read()
	if event is None or event == "Exit":
		cv2.destroyAllWindows()
		break


	if event == '显示明场图像':
		try:
			IMG = cv2.imread(values['_IMGDIR_'],0)
			IMG = cv2.equalizeHist(IMG)
			plt.figure()
			plt.imshow(IMG,'gray'), plt.title('Bright field image')
			plt.show(block=False)
			#Updating('显示明场图像')
		except Exception as e:
			#Updating('无法打开明场图像')
			pass

	if event == '显示荧光图像':
		try:
			FIMG = cv2.imread(values['_FIMGDIR_'],0)
			plt.figure()
			plt.imshow(FIMG, 'gray'), plt.title('Fluorescent image')
			plt.show(block=False)
			Updating('显示荧光图像')
		except Exception as e:
			Updating('无法显示荧光图像')
			pass

	if event == '开始识别':
		#Updating('开始识别液滴')
		print('开始识别液滴')
		try:
			#Updating('圆检测...')
			CircleDetection()
			print('圆检测...')
			try:
				DrawCircles()
			except Exception as e:
				#Updating('圆检测失败')
				print('圆检测失败')
				pass
		except Exception as e:
			#Updating('圆检测失败')
			print('圆检测失败')
			pass

	#if event == 'Close Data Figure':
	#	cv2.destroyAllWindows()
	if event == '开始分析':
		try:
			analysis()
			window.Element('_OUTPUT_').update('Analysing..')
		except Exception as e:
			window.Element('_OUTPUT_').update('Something is wrong.')
			pass
	
	if event == '区分液滴':
		try:
			CellNumCount()
		except Exception as e:
			window.Element('_OUTPUT_').update('Cannot count cells.')
			pass
	
	if event == 'Run by default':
		try:
			CircleDetection()
			try:
				DrawCircles()
				try:
					analysis()
				except Exception as e:
						window.Element('_OUTPUT_').update('Analysising failed.')
			except Exception as e:
				window.Element('_OUTPUT_').update('Cannot draw the circles.')
				pass
		except Exception as e:
			pass

	if event == '显示液滴':
		try:
			ShowDrops(-1)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong with showing the drops')
			pass

	if event == '显示无细胞液滴':
		try:
			ShowDrops(0)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong with showing the drops')
			pass

	if event == '显示单细胞液滴':
		try:
			ShowDrops(1)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong with showing the drops')
			pass

	if event == '显示多细胞液滴':
		try:
			ShowDrops(2)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong with showing the drops')
			pass

	if event == '总荧光分布直方图':
		try:
			ShowHist(-1)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong with showing the histgram')
			pass

	if event == '无细胞荧光分布直方图':
		try:
			ShowHist(0)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong with showing the histgram')
			pass

	if event == '单细胞荧光分布直方图':
		try:
			ShowHist(1)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong with showing the histgram')
			pass

	if event == '多细胞荧光分布直方图':
		try:
			ShowHist(2)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong with showing the histgram')
			pass
	
	if event == '导出校正后荧光图像':
		try:
			cv2.imwrite('FImagewithIC.bmp',fimg)
		except Exception as e:
			window.Element('_OUTPUT_').update('Something went wrong')
			pass

	if event == '显示校正效果':
		try:
			plt.figure()
			plt.subplot(221),plt.imshow(imgo,'gray'),plt.title('Raw Bright Image')
			plt.subplot(222),plt.imshow(fimgo,'gray'),plt.title('Raw Fluorescent Image')
			plt.subplot(223),plt.imshow(img,'gray'),plt.title('Bright Image with IC')
			plt.subplot(224),plt.imshow(fimg,'gray'),plt.title('Fluorescent Image with IC')
			plt.show(block=False)
		except Exception as e:
			window.Element('_OUTPUT_').update('Cannot show the images.')
			pass 


	if event == 'Export':
		try:
			timestr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
			savingpath = values['_EXPORT_']
			ff = timestr+'.xls'
			filename = os.path.join(savingpath,ff)
			Export2Excel(filename)
		except Exception as e:
			window.Element('_OUTPUT_').update('Aha! Some work still needs to be done.')
		pass 


window.Close()

#sg.Popup('The GUI returned:', event, values)
