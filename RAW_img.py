import os
import rawpy # RAW file processor - wrapper for libraw / dcraw
import inspect
import numpy as np
import math as m
import pandas as pd
import statistics as stats
import exiftool
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D      
import sys


"""---------------------------------------------------------------------------------------------------------------------------------------------------"""

class Raw_img():




	def __init__(self,img_loc,filename,ext = '.ARW'):
		""" Import the raw file using rawpy
		img_loc = str - relative to this file...I think
		filename = str - as explained
		ext = str - file extension - default is .ARW but can do .JPG """

		# Status of different functions
		self.status = {'undistorted': False , 
			'normalised' : False , 
			'grayscale' : False , 
			'black_level' : False , 
			'cropped' : False ,
			'aligned' : False}

		# Change working directory to the directory of this script
		os.chdir(os.path.dirname(__file__)) 
		self.file_path = img_loc + filename + ext
		
		# make inputs attributes
		self.img_loc = img_loc
		self.filename = filename
		self.ext = ext[1:]



		if ext == '.ARW':
			x = rawpy.imread(self.file_path) # raw file is imported using rawpy
			self.raw_image = x.raw_image
			x.close()
		elif ext == '.JPG':
			self.raw_image = mpimg.imread(os.path.join(os.path.dirname(self.file_path), filename + ext))
		# Split into rgb channels
		self.rgb_channels(ext)
		# Get metadata
		self.get_metadata()
		# Get sizes
		self.get_size()
		
		
		



	def get_metadata(self):
		"""Get Image Metadata and clean"""
		self.metadata = {}
		with exiftool.ExifTool() as et: 
			md = et.get_metadata(self.file_path)
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
		# - Time image captured
		

	def get_size(self):
		"""Get size of image"""
		self.width = self.raw_image.shape[1]
		self.height = self.raw_image.shape[0]



	def rgb_channels(self,ext = '.ARW'):
		
		""" Create Red, Green and Blue Arrays
		ext = str. file extension default is raw file, can also have .JPG """
		if ext == '.ARW':
			self.raw_red = self.raw_image[::2, ::2]
			self.raw_green = np.array( ( (self.raw_image[::2,1::2] + self.raw_image[1::2, ::2] ) / 2).round(), dtype = np.uint16)
			self.raw_blue = self.raw_image[1::2,1::2]
		elif ext == '.JPG':
			self.raw_red = self.raw_image[:,:,0]
			self.raw_green = self.raw_image[:,:,1]
			self.raw_blue = self.raw_image[:,:,2]
		#print('self.red (' + str(self.red.shape) + ' self.green ' + str(self.green.shape)  
		#+' self.blue ' + str(self.blue.shape) +' successfully created')



	
	def save_histogram(self, crop = True):
		"""Creates and saves a histogram of the image to the same folder as the image"""
		try:
			colors = ['red','green','blue']
			hist_col = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
			fig = plt.figure()
			bits = int(self.metadata['BitsPerSample'])
			for C in colors:
				ax = fig.add_subplot(3,1,colors.index(C)+1)
				print('Plotting ' + C + ' channel')
				if crop == False:
					if not os.path.exists(self.img_loc + self.ext + '/raw_hist/'):
						os.makedirs(self.img_loc + self.ext + '/raw_hist/')

					ax.hist(getattr(self,'raw_' + C).reshape(-1), bins = (2**bits) , range=(0, 2**bits+1), color = hist_col[colors.index(C)])
					hist_name = self.img_loc + self.ext + '/raw_hist/' + self.filename + '.png'

				else:
					
					if not os.path.exists(self.img_loc + self.ext + '/hist/'):
						os.makedirs(self.img_loc + self.ext + '/hist/')

					ax.hist(getattr(self,C).reshape(-1), bins = (2**bits), range=(0, 2**bits+1), color = hist_col[colors.index(C)])
					hist_name = self.img_loc + self.ext + '/hist/' + self.filename + '.png'
				plt.title(C)
				# Save Histogram				
			fig.savefig(hist_name)
			plt.close()
		except AttributeError as e:
			print(str(e)+' - need to run crop_img to get histogram!')


	def choose_crop(self):
		'''method is allows user to return a suitable crop area'''

		# Show an image in interactive mode
		plt.ion()
		ax = plt.axes()
		if self.ext == 'JPG':
			if self.status['cropped'] == False: # cropping the original image
				ax.imshow(self.raw_image)
			else:  # cropping again for analysis area
				ax.imshow(self.image)
		else:
			if self.status['cropped'] == False:  # cropping the original image
				ax.imshow(self.raw_red)
			else: # cropping again for analysis area
				ax.imshow(self.image)

		response = 'no'
		while 'y' not in response.lower():
			
			# input position of the crop        
			xy = []
			xy.append(int(input('x-coordinate of the top left corner: ')))
			xy.append(int(input('y-coordinate of the top left corner: ')))
			width = int(input('x-coordinate of the bottom right corner: ')) - xy[0]
			height = int(input('y-coordinate of the bottom right corner: ')) - xy[1]

			# display crop on image
			rect = patches.Rectangle( (xy[0], xy[1]), width, height, linewidth = 1, edgecolor='r', facecolor = 'none')
			ax.add_patch(rect)
			plt.draw()
			response = input('Are you happy with the crop area?  Do you want to continue? [Y/N]')
		
		# End interactive mode and close figure
		plt.ioff()
		plt.close()
		# return to crop information
		return (xy, width, height)


	def crop_img(self, crop_pos):
		"""Crops the image to the space you want, if check_crop = True, the image will be displayed 
		and you have the option of re aligning if you want """
		
		# Input the 
		self.crop_xy = crop_pos[0]
		self.crop_width = crop_pos[1]
		self.crop_height = crop_pos[2]

		# Make the crops
		self.image = self.raw_image[self.crop_xy[1]:(self.crop_xy[1] + self.crop_height) , self.crop_xy[0]: (self.crop_xy[0] + self.crop_width)]
		self.red = self.raw_red[self.crop_xy[1]:(self.crop_xy[1] + self.crop_height) , self.crop_xy[0]: (self.crop_xy[0] + self.crop_width)]
		self.green = self.raw_green[self.crop_xy[1]:(self.crop_xy[1] + self.crop_height) , self.crop_xy[0]: (self.crop_xy[0] + self.crop_width)]
		self.blue= self.raw_blue[self.crop_xy[1]:(self.crop_xy[1] + self.crop_height) , self.crop_xy[0]: (self.crop_xy[0] + self.crop_width)]
		# housekeeping
		self.status['cropped'] = True



	def disp_img(self, disp = True, crop = False, save = False, channel = 'red', colormap = 'gray'):
		"""Function displays the image on the screen
		OPTIONS - 	disp - True - whether to actually display the image or not
					crop = True - cropped as by crop_img False - Full image
					save - False - save one of the channels
					channel = string - red, green, blue
					colormap - control the colors of the image - default is grayscale"""
		
		
		if disp == True:
			if crop == False:
				plt.imshow(getattr(self,'raw_' + channel), aspect = 'equal', cmap = colormap)
			else:
				plt.imshow(getattr(self, channel), aspect = 'equal', cmap = colormap)
			plt.axis('off')
			plt.title(channel.capitalize()+ ' channel')
		
		if save == True:
			if crop == False:
				# Create a folder with name if doesn't exist
				if not os.path.exists(self.img_loc + self.ext + '/raw_' + channel + '_channel/'):
					os.makedirs(self.img_loc + self.ext + '/raw_' + channel + '_channel/')
				# save image
				plt_name = self.img_loc + self.ext + '/raw' + channel + '_channel/' + self.filename + '.png'
				plt.imsave(plt_name,getattr(self, 'raw_' + channel), cmap = colormap)
			else:
				if not os.path.exists(self.img_loc + self.ext + '/' + channel + '_channel/'):
					os.makedirs(self.img_loc + self.ext + '/' + channel + '_channel/')
				# save image
				plt_name = self.img_loc + self.ext + '/' + channel + '_channel/' + self.filename + '.png'
				plt.imsave(plt_name,getattr(self, channel), cmap = colormap)



	def black_offset(self, method = 0, *blk_imgs):
		"""0 intensity does not normally represent black so we need to offset the image by the amount required
		We have 2 methods of doing that
		method = 0 (default) use the metadata in the image 
		method = 1 use 1 or a series of black images"""
		if self.status['cropped'] == False:
			sys.exit('Need to crop the image first')
		if self.status['black_level'] == True:
			sys.exit('Black offset already applied to this image')

		if method == 0 :
			Black_Level = np.array(list(map(int,self.metadata['BlackLevel'].split()))).mean()
			self.raw_image = np.subtract(self.raw_image, np.ones_like(self.raw_image)* Black_Level)
			self.red = np.subtract(self.red,  np.ones_like(self.red)*Black_Level)
			self.green = np.subtract(self.green,  np.ones_like(self.green)* Black_Level)
			self.blue = np.subtract(self.blue,  np.ones_like(self.blue)* Black_Level)
		elif method == 1:
			sys.exit('Code hasn''t yet been written!')

		self.status['black_level'] = True

	def undistort(self):
		self.status['undistorted'] = True

	def normalise(self, bg_img):
		'''method will subtract a background image from a the class instance.  
		Subtraction only happens on the full images
		bg_img = np.array of rgb channels, same size as the images'''
		# Check we aren't normalising anything we shouldn't
		#if self.status['undistorted'] != True or any bg_img.status['undistorted'] != True:
			#sys.exit('Image needs to be undistorted before normalising')
		if self.status['normalised'] == True:
			sys.exit('Image has already been normalised')
		if self.status['cropped'] == False:
			sys.exit('You should crop the image before normalising')
		if (self.status['black_level'] == False) and (self.ext == 'ARW') :
			sys.exit('You should offset the by the black level before normalising')
		
		# divide by the background image

		self.red = np.divide(self.red , bg_img[0])
		self.green = np.divide(self.green , bg_img[1])
		self.blue = np.divide(self.blue , bg_img[2])
		
		# housekeeping
		self.status['normalised'] = True



#- Calibrate image via camera calibration
#- Black Level offset
#- undistort image
#- Normalise to background image
#- Re align images
#- convert pixels to real life scale
#- Calculate dye concentration and density fields """



	def define_analysis_strips(self, analysis_area, strip_width, channel = 'red', display = False):
		'''defines an area of processed image of channel ... to analyse.
		returns a dictionary of strip section and the np.array sitting in value
		img = RAW_img class object
		analysis_area = tuple from choose_crop() total area to analyse
		strip_width = int in pixels'''
		number_of_strips = m.floor(analysis_area[1] / strip_width) # find maximum number of stips based on spacing and width
		x1 = analysis_area[0][0]
		y1 = analysis_area[0][1]
		# x,y bottom right corner
		x2 = analysis_area[0][0] + number_of_strips*strip_width 
		y2 = analysis_area[0][1] + analysis_area[2] # y-coordinate
		strip_interfaces = [int(i) for i in np.linspace(analysis_area[0][0], x2, number_of_strips+1)]
		strip_label = range(number_of_strips) # counter for the strips
		if display == True:
			plt.ion()
			ax = plt.axes()
			ax.imshow(getattr(self, channel))
			rect1 = patches.Rectangle( (x1, y1), (x2-x1), (y2-y1), linewidth = 1, edgecolor='r', facecolor = 'none')
			ax.add_patch(rect1)
			j = 0
			for i in strip_interfaces[:-1]:
				l = Line2D( [i,i], [y1,y2] , linewidth = 1, color = 'r')
				ax.add_line(l)
				plt.text( i + round(strip_width/2) , stats.mean([y1,y2]) , str(strip_label[j]), color = 'r' )
				plt.draw()
				j += 1
			plt.ioff()
			if not os.path.exists(self.img_loc + 'analysis/'):
				os.makedirs(self.img_loc + 'analysis/')
			plt.savefig(self.img_loc + 'analysis/' + channel + '_channel_analysis_strips.png')
			plt.close()

		self.strips =  { str(strip_label[i]) : getattr(self,channel)[ analysis_area[0][1]:y2 , strip_interfaces[i] : strip_interfaces[i+1] ] for i in strip_label} 



	def one_d_density(self, n = 10):
		'''takes in a dictionary containing np.arrays (strips),
		produces plot, or horizontally average values
		smoothness = number of pixels to do a moving average '''
		self.rho = pd.DataFrame(columns = self.strips.keys())

		for k, v in self.strips.items():
			# horizontal mean of each strip
			self.rho[k] = pd.Series(np.mean(v, axis = 1))

			# smoothed out noise with moving average
			self.rho[k + '_' + str(n)]= self.rho[k].rolling( n, min_periods = 1, center = True ).mean()
			plt.plot(self.rho[k],np.arange(len(self.rho[k]) ), label = k )
		plt.legend()
			# Save a figure
		plt.savefig(self.img_loc + 'analysis/rho_profile_' + self.filename + '.png')
		plt.close()



		
	




	
