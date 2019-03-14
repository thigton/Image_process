import os
import rawpy # RAW file processor - wrapper for libraw / dcraw
import inspect
import numpy as np
import exiftool
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import sys


"""---------------------------------------------------------------------------------------------------------------------------------------------------"""

class Raw_img():

	"""Status of different functions"""
	status = {'undistorted': False , 
			'normalised' : False , 
			'grayscale' : False , 
			'black_level' : False , 
			'cropped' : False ,
			'aligned' : False}


	def __init__(self,img_loc,filename):
		os.chdir(os.path.dirname(__file__)) # Change working directory to the directory of this script
		""" Import the raw file using rawpy """
		self.raw_file_path = img_loc + filename + '.ARW'

		self.img_loc = img_loc
		self.filename = filename
		x = rawpy.imread(self.raw_file_path) # raw file is imported using rawpy

		self.raw_image = x.raw_image
		# Get metadata
		self.get_metadata()
		#Import jpeg
		#self.jpg = mpimg.imread(os.path.join(os.path.dirname(img_loc + filename + '.JPG'), filename + '.JPG'))
		# Split into rgb channels
		self.rgb_channels()
		# Get sizes
		self.get_size(x)
		x.close()

	def get_metadata(self):
		"""Get Image Metadata and clean"""
		self.metadata = {}
		with exiftool.ExifTool() as et: 
			md = et.get_metadata(self.raw_file_path)
		for key in md.keys():
			# remove the text before the colon in the keys
			new_key = key.split(':')[-1]
			self.metadata[new_key] = md[key]
		del(md)
		# Important key values
		# - BitsPerSample
		# - ISO
		# - FocalLength
		# - ShutterSpeed
		# - Aperture
		# - Make
		# - Model
		# - DistortionCorrectionSetting
		# - BlackLevel
		

	def get_size(self,x):
		"""Get size of image"""
		self.width = x.sizes.raw_width
		self.heigth = x.sizes.raw_height



	def rgb_channels(self):
		""" Create Red, Green and Blue Arrays """
		self.red = self.raw_image[::2, ::2]
		self.green = np.array(((self.raw_image[::2,1::2] + self.raw_image[1::2, ::2] ) / 2).round(), dtype = np.uint16)
		self.blue = self.raw_image[1::2,1::2]
		#print('self.red (' + str(self.red.shape) + ' self.green ' + str(self.green.shape)  
		#+' self.blue ' + str(self.blue.shape) +' successfully created')



	
	def save_histogram(self, crop = True):
		"""Creates and saves a histogram of the image to the same folder as the image"""
		try:
			colors = ['red','green','blue']
			hist_col = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
			fig = plt.figure()
			
			for C in colors:
				ax = fig.add_subplot(3,1,colors.index(C)+1)
				print('Plotting ' + C + ' channel')
				if crop == False:
					ax.hist(getattr(self,C).reshape(-1), bins = (2**14) , range=(0, 2**14+1), color = hist_col[colors.index(C)])
					hist_name = self.img_loc + 'Hist' + self.filename + '.png'
				elif crop == True:
					C_crop = 'crop_' + C
					ax.hist(getattr(self,C_crop).reshape(-1), bins = (2**14) , range=(0, 2**14+1), color = hist_col[colors.index(C)])
					hist_name = self.img_loc + 'Hist_crop_' + self.filename + '.png'
				plt.title(C)
				# Save Histogram				
			fig.savefig(hist_name)
			print('Histogram successfully saved as ' + hist_name)
		except AttributeError as e:
			print(str(e)+' - need to run crop_img to get histogram!')





	def crop_img(self, xy , width, height, check_crop = False, save_red = False, save_green = False):
		"""Crops the image to the space you want, if check_crop = True, the image will be displayed 
		and you have the option of re aligning if you want """
		#if self.status['undistorted'] != True:
		#	sys.exit('Image needs to be undistorted before cropping')
		#if self.status['cropped'] != False:
		#	response = input('Image has already been cropped, do you want to overwrite? [Y/N]')
		#	if 'y' not in response.lower():
		#		print('continue with existing cropped image...')
		#		return
		#if self.status['black_level'] != True:
		#	sys.exit('Image should be black-offset before cropping')
		
		# Input the 
		self.crop_xy = xy
		self.crop_width = width
		self.crop_height = height
		if check_crop == True:
			plt.ion()
			ax = plt.axes()
			ax.imshow(self.red)
			rect = patches.Rectangle( (self.crop_xy[0], self.crop_xy[1]), self.crop_width, self.crop_height, linewidth = 1, edgecolor='r', facecolor = 'none')
			ax.add_patch(rect)
			response = input('Are you happy with the crop area?  Do you want to continue? [Y/N]')
			if 'y' in response.lower():
				self.crop_img(xy, width, height, check_crop == False)
			else:
				while 'y' not in response.lower():
					XY = []
					XY.append(int(input('x-coordinate of the top left corner: ')))
					XY.append(int(input('y-coordinate of the top left corner: ')))
					WIDTH = int(input('Width of crop: '))
					HEIGHT = int(input('height of crop: '))
					rect.remove()
					rect = patches.Rectangle( (XY[0], XY[1]), WIDTH, HEIGHT, linewidth = 1, edgecolor='r', facecolor = 'none')
					ax.add_patch(rect)
					plt.draw()
					response = input('Are you happy with the crop area?  Do you want to continue? [Y/N]')
				self.crop_img(XY, WIDTH, HEIGHT, check_crop = False)
		else:
			# Make the crops
			self.crop_raw_image = self.raw_image[self.crop_xy[1]:(self.crop_xy[1] + self.crop_height) , self.crop_xy[0]: (self.crop_xy[0] + self.crop_width)]
			self.crop_red = self.red[self.crop_xy[1]:(self.crop_xy[1] + self.crop_height) , self.crop_xy[0]: (self.crop_xy[0] + self.crop_width)]
			self.crop_green = self.green[self.crop_xy[1]:(self.crop_xy[1] + self.crop_height) , self.crop_xy[0]: (self.crop_xy[0] + self.crop_width)]
			self.crop_blue= self.blue[self.crop_xy[1]:(self.crop_xy[1] + self.crop_height) , self.crop_xy[0]: (self.crop_xy[0] + self.crop_width)]
			print('Crop successful')
			self.status['cropped'] = True

		"""OPTION: Save the red channel to png to view"""
		# Save red channel to file
		if save_red == True:
			self.disp_img(disp = False, save = True, channel = 'red')
		if save_green == True:
			self.disp_img(disp = False, save = True, channel = 'red')

			
	def disp_img(self, disp = True, crop = False, save = False, channel = 'red', colormap = 'gray'):
		"""Function displays the image on the screen
		OPTIONS - 	disp - True - whether to actually display the image or not
					crop = True - cropped as by crop_img False - Full image
					save - False - save one of the channels
					channel = string - red, green, blue
					colormap - control the colors of the image - default is grayscale"""
		
		fig = plt.figure()
		if crop == False:
			plt.imshow(getattr(self,channel), aspect = 'equal', cmap = colormap)
		else:
			crop_channel = 'crop_' + channel
			plt.imshow(getattr(self, crop_channel), aspect = 'equal', cmap = colormap)
		plt.axis('off')
		plt.title(channel.capitalize()+ ' channel')
		
		if save == True:
			if not os.path.exists(self.img_loc + channel + '_channel/'):
				os.makedirs(self.img_loc + channel + '_channel/')
			plt_name = self.img_loc + channel + '_channel/' + self.filename + '.png'
			fig.savefig(plt_name)
		if disp == True:
			plt.show()
		del(fig)


	def black_offset(self, method = 0, *blk_imgs):
		"""0 intensity does not normally represent black so we need to offset the image by the amount required
		We have 2 methods of doing that
		method = 0 (default) use the metadata in the image 
		method = 1 use 1 or a series of black images"""
		if method == 0 :
			Black_Level = np.array(list(map(int,self.metadata['BlackLevel'].split()))).mean()
			self.raw_image = np.subtract(self.raw_image, np.ones_like(self.raw_image)* Black_Level)
			self.red = np.subtract(self.red,  np.ones_like(self.red)* Black_Level)
			self.green = np.subtract(self.green,  np.ones_like(self.green)* Black_Level)
			self.blue = np.subtract(self.blue,  np.ones_like(self.blue)* Black_Level)
		elif method == 1:
			sys.exit('Code hasn''t yet been written!')

		self.status['black_level'] = True

	def undistort(self):
		self.status['undistorted'] = True

	def normalise(self, *bg_imgs):
		pass
		# Check we aren't normalising anything we shouldn't
		#if self.status['undistorted'] != True or any bg_img.status['undistorted'] != True:
			#sys.exit('Image needs to be undistorted before normalising')

		


#- Calibrate image via camera calibration
#- Black Level offset
#- undistort image
#- Normalise to background image
#- Re align images
#- convert pixels to real life scale
#- Calculate dye concentration and density fields """


#print(dir(rawpy))
#print(dir(rawpy.Rawpy))