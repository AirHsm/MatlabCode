import PySimpleGUI as sg
import cv2
import numpy as np
import matplotlib.pyplot as plt


# set theme
sg.theme('BluePurple')

# callback functions

def DropsDetection(values):
	#imgdir, fimgdir, minr, maxr
	global dropnum
	global circles
	global img 
	global fimg 

	img = cv2.imread(values['_IMGDIR_'],0)
	fimg = cv2.imread(values['_FIMGDIR_'],0)

	minr = int(values['_MINRADIUS_'])
	maxr = int(values['_MAXRADIUS_'])

	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 100, minRadius=minr, maxRadius=maxr)
	circles = np.uint16(np.around(circles))

	dropnum = circles.shape[1]


def analysis():

	global DropDensity
	global DropSTD 
	global Drops
	global dropsnumef

	DropDensity = []
	DropSTD = []
	Drops = []

	m, n = img.shape

	x = np.linspace(0, n-1, n)
	y = np.linspace(0, m-1, m)

	X,Y = np.meshgrid(x, y)


	for i in range(dropnum):
		centx = int(circles[0, i, 0])
		centy = int(circles[0, i, 1])
		r = int(circles[0, i, 2])+1
		t = ((centx-r)>1)*((centx+r)<n)*((centy-r)>1)*((centy+r)<m)
		if t:
				Rv = np.sqrt((X-centx)**2 + (Y-centy)**2)

				drop = fimg[centy-r:centy+r,centx-r:centx+r]

				Drops.append(drop)
				DropSTD.append(np.std(drop))
				DropDensity.append(np.sum(fimg*(Rv<=r))/r**2)

	dropsnumef = np.max(np.shape(Drops))

	plt.figure()
	plt.subplot(131), plt.imshow(Drops[1]), plt.xticks([]), plt.yticks([])
	plt.subplot(132)
	plt.hist(DropDensity),plt.title('Fluorescence density')
	plt.subplot(133)
	plt.hist(DropSTD), plt.title('Fluorescence STD')
	plt.show(block=False)


def DrawCircle():

	cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cfimg = cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR)

	for i in circles[0,:]:
		cv2.circle(cimg, (i[0],i[1]), i[2], (255,0,0),2)
		cv2.circle(cimg, (i[0],i[1]), 2, (255,0,255),5)
		cv2.circle(cfimg, (i[0],i[1]), i[2], (255,0,0),2)

	cv2.imshow('Bright field image', cimg)
	cv2.imshow('Fluorescent image with detected circles', cfimg)


def CellNumCount():

	# global Drops

	global DropsWithSingleCell
	global DropsWithoutCell
	global DropsWithMultiCell

	global DropsWithSingleCellNum
	global DropsWithoutCellNum
	global DropsWithMultiCellNum



	DropsWithSingleCell = []
	DropsWithoutCell = []
	DropsWithMultiCells =[]



	for drop in Drops:
		ret, thresh = cv2.threshold(drop, 127, 255, 0)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		contourNum = hierarchy.shape[1]

		if contourNum == 0:
			DropsWithoutCell.append(drop)
		if contourNum == 1:
			rect = cv2.minAreaRect(contours[0])
			WH = rect[1]
			if np.max(WH)/np.min(WH) > 1.3:
				DropsWithMultiCells.append(drop)
			else:
				DropsWithSingleCell.append(drop)
		if contourNum > 1:
			DropsWithMultiCells.append(drop)

	DropsWithSingleCellNum = np.max(np.shape(DropsWithSingleCell))
	DropsWithoutCellNum = np.max(np.shape(DropsWithoutCell))
	DropsWithMultiCellsNum = np.max(np.shape(DropsWithMultiCells))


# layout

layout = [
			[sg.Text('明场图像位置:'), sg.Input('/Users/hushiming/workspace/temp/HYZ/10x-1.tif', key='_IMGDIR_'), sg.FileBrowse(target='_IMGDIR_')],
			[sg.Text('荧光图像位置:'), sg.Input('/Users/hushiming/workspace/temp/HYZ/10x-2.tif', key='_FIMGDIR_'), sg.FileBrowse(target='_FIMGDIR_')],
			[sg.Text('液滴半径范围 (pixel):'),sg.Text('Min:', size=(5,1)), sg.Input('58', key='_MINRADIUS_', size=(5,1)), sg.Text('Max:', size=(5,1)), sg.Input('62', key='_MAXRADIUS_', size=(5,1))], 
			[sg.Button('Start'), sg.Button('Close Data Figure'), sg.Button('Analysis'), sg.Button('Cell Count'), sg.Exit()],
			[sg.Text('液滴总数:'), sg.Text('00000', key='_DROPNUM_')],
			[sg.Text('Output:', size=(3,2)), sg.Multiline(key='_OUTPUT_', size=(60,2))],
]

window = sg.Window('液滴检测', layout)

while True:
	event, values = window.read()
	if event is None or event == "Exit":
		cv2.destroyAllWindows()
		break
	if event == 'Start':
		try:
			DropsDetection(values)
			DrawCircle()
			OutputStr = ['液滴总数：' + str(dropnum)]				
			window.Element('_OUTPUT_').update(OutputStr)
		except Exception as e:
			window.Element('_OUTPUT_').update('Cannot find any circles.\nPlease check the input data or radius range of the drops.')
			pass

	if event == 'Close Data Figure':
		cv2.destroyAllWindows()
	if event == 'Analysis':
		analysis()
	if event == 'Cell Count':
		try:
			CellNumCount()
		except Exception as e:
			pass
			
		

		


window.Close()

#sg.Popup('The GUI returned:', event, values)