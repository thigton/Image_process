import glob
import os
import rawpy # RAW file processor - wrapper for libraw / dcraw
import numpy as np
import math as m
import pandas as pd
import exiftool
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D 
import matplotlib.gridspec as gridspec     
import sys
import csv
from scipy.signal import medfilt
import itertools

"""---------------------------------------------------------------------------------------------------------------------------------------------------"""
#pylint: disable=no-member
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
		# Get sizes 
		self.get_size()

		
		

	def get_metadata(self):
		"""Get Image Metadata and clean"""
		metadata = {}
		with exiftool.ExifTool() as et: 
			md = et.get_tags(['BitsPerSample', 'ISO', 'ShutterSpeed', 'Aperture','Make','Model','BlackLevel' ],self.file_path)
			for key in md.keys():
				# remove the text before the colon in the keys
				new_key = key.split(':')[-1]
				metadata[new_key] = md[key]
			del(md)
			et.terminate()
		return metadata

	def get_time(self,t0 = 0):
		with exiftool.ExifTool() as et:
			self.time = int(datetime.strptime(et.get_tag('ModifyDate',self.file_path), '%Y:%m:%d %H:%M:%S').timestamp() - t0 )
			et.terminate()

	def get_experiment_conditions(self):
		'''Accesses csv files with experiment details and make them attributes
		exp_no - str this should match a reference in the csv file to determine which row to read in.'''
		 # grab folder name which ids the experiment

		with open('Data/experiment_details.csv','r') as csvfile:

			reader = csv.reader(csvfile, delimiter = ',')
			for row in reader:
				if row[10] == self.img_loc.split('/')[-2]:
					self.bottom_opening_diameter = int(row[3])
					self.side_opening_height = int(row[4])
					self.sol_no = row[5]

		with open('Data/solution.csv','r') as csvfile:
		 	reader = csv.reader(csvfile, delimiter = ',')
		 	for row in reader:
				 if row[0] == self.sol_no:
					 self.sol_denisty = float(row[5])
		

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



	
	def save_histogram(self, metadata, crop = True):
		"""Creates and saves a histogram of the image to the same folder as the image"""
		try:
			colors = ['red','green','blue']
			hist_col = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
			fig = plt.figure()
			bits = int(metadata['BitsPerSample'])
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


	def choose_crop(self, **kwargs):
		'''method is allows user to return a suitable crop area'''

		# Show an image in interactive mode
		plt.ion()
		ax = plt.axes()
		if self.ext == 'JPG':
			if self.status['cropped'] == False: # cropping the original image
				ax.imshow(self.raw_image)
			else:  # cropping again for analysis area
				ax.imshow(self.image)
		else: #if raw file
			if self.status['cropped'] == False:  # cropping the original image
				ax.imshow(self.raw_red)
			else: # cropping again for analysis area
				ax.imshow(self.image)

		response = 'no'
		while 'y' not in response.lower():

			# input position of the crop        
			x1 = int(input('x-coordinate of the top left corner: '))
			y1 = int(input('y-coordinate of the top left corner: '))
			width = int(input('x-coordinate of the bottom right corner: ')) - x1
			height = int(input('y-coordinate of the bottom right corner: ')) - y1
			# display crop on image
			rect = patches.Rectangle( (x1, y1), width, height, linewidth = 1, edgecolor='r', facecolor = 'none')
			ax.add_patch(rect)
			plt.draw()
			response = input('Are you happy with the crop area?  Do you want to continue? [Y/N]')
			rect.set_visible = False
		# End interactive mode and close figure
		plt.ioff()
		plt.close()
		# return to crop information as dictionary
		return {'x1': x1, 'y1' : y1,  'width' : width, 'height' : height}


	def crop_img(self, crop_pos):
		"""Crops the image to the space you want. Based on predefines crop coordinates """
		
		# Input the 
		self.crop_x1 = crop_pos['x1']
		self.crop_y1 = crop_pos['y1']
		self.crop_width = crop_pos['width']
		self.crop_height = crop_pos['height']

		# Make the crops
		self.image = self.raw_image[self.crop_y1:(self.crop_y1 + self.crop_height) , self.crop_x1: (self.crop_x1 + self.crop_width)]
		self.red = self.raw_red[self.crop_y1:(self.crop_y1 + self.crop_height) , self.crop_x1: (self.crop_x1 + self.crop_width)]
		self.green = self.raw_green[self.crop_y1:(self.crop_y1 + self.crop_height) , self.crop_x1: (self.crop_x1 + self.crop_width)]
		self.blue= self.raw_blue[self.crop_y1:(self.crop_y1 + self.crop_height) , self.crop_x1: (self.crop_x1 + self.crop_width)]
		# housekeeping
		self.status['cropped'] = True

	def disp_img(self, disp = True, crop = False, save = False, channel = 'red', colormap = 'inferno'):
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
				plt.imshow(getattr(self, channel), aspect = 'equal', cmap = colormap, vmin = 0, vmax = 1)
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
				plt.imsave(plt_name,getattr(self, channel), cmap = colormap, vmin = 0, vmax = 1)

	def convert_centre_pixel_coordinate(self,crop_pos):
		'''returns the new coordinates of the centre point of the image based on the crop in crop_pos'''
		x = self.width/2 - crop_pos['x1']
		y = self.height/2 - crop_pos['y1']
		self.centre = (x,y)


	def black_offset(self,metadata, method = 0, *blk_imgs):
		"""0 intensity does not normally represent black so we need to offset the image by the amount required
		We have 2 methods of doing that
		method = 0 (default) use the metadata in the image 
		method = 1 use 1 or a series of black images"""
		if self.status['cropped'] == False:
			sys.exit('Need to crop the image first')
		if self.status['black_level'] == True:
			sys.exit('Black offset already applied to this image')

		if method == 0 :
			Black_Level = np.array(list(map(int,metadata['BlackLevel'].split()))).mean()
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



	def define_analysis_strips(self, crop_pos, vertical_scale, door_strip_width = 100, channel = 'red', save = False):
		'''defines an area of processed image of channel ... to analyse.
		returns pd.dataframes of the door strip and the box strip
		img = RAW_img class object
		crop_pos = dictionary from choose_crop() total area to analyse
		door_strip_width = int , number of pixels to analyse seperately close to the door to see if there is a difference.
		channel = str, rgb channel
		save = bool, do you want to save an image of the sections'''

		width = crop_pos['width']
		height = crop_pos['height']
		x1 = crop_pos['x1']
		y1 = crop_pos['y1']

		# x2 is the x-coordinate of the interface between the dor strip and the rest
		x2 = x1 + door_strip_width
		# y-coordinate of the bottom of the 
		y2 = y1 + height 

		def transmit_to_absorb(df):
			'''convert the element in the dataframe from Transmittance to Absorbance using the beer lambert law'''
			return df.applymap(lambda x : 0-np.log(x))

		# convert analysis area to Absorbance
		analysis_array = transmit_to_absorb(pd.DataFrame(getattr(self, channel)[y1:y2, x1:+width]))

		# Split into analysis areas based on front and back scale and the two strips
		front_scale = pd.Series(vertical_scale[0] , name = 'h/H')
		back_scale = pd.Series(vertical_scale[1] , name = 'h/H')
		for scale , strip in itertools.product(['front', 'back'], ['box', 'door']):
			use_scale = back_scale if scale == 'back' else front_scale
			range_tup = (x2-x1, width-x1) if strip == 'box' else (0, x2-x1)
			setattr(self, scale + '_' + strip + '_strip' , analysis_array[ [x for x in range(range_tup[0], range_tup[1])] ] )
			getattr(self, scale + '_' + strip + '_strip').set_index(use_scale, inplace = True)

		



		if save == True:
			plt.ion()
			ax = plt.axes()
			ax.imshow(getattr(self, 'image'))
			rect1 = patches.Rectangle( (x1, y1), width, height, linewidth = 1, edgecolor='r', facecolor = 'none')
			ax.add_patch(rect1)
			ax.add_line(Line2D([x2,x2],[y1,y2],linewidth = 1, color = 'r'))
			plt.text( (x1+x2)/2 , (y1+y2)/2 , 'door strip', color = 'r', rotation = 90)
			plt.text( (x2+x1+width)/2 , (y1+y2)/2 , 'box strip', color = 'r', rotation = 90)
			plt.draw()
			plt.ioff()
			if not os.path.exists(self.img_loc + 'analysis/'):
				os.makedirs(self.img_loc + 'analysis/')
			plt.savefig(self.img_loc + 'analysis/' + channel + '_channel_analysis_strips.png')
			plt.close()


		

	def one_d_density(self, vertical_scale, save_fig = False):
		'''finds horizontal average and standard deviation of box_strip and door_strip) 
		and appends dataframe to a csv file containing this information for all images in the experiment.'''
		
		# initialise dataframe with h/H data in place
		columns = pd.MultiIndex.from_product([[self.time] , ['door', 'box'], ['mean','std'] ], names = ['time','data','attribute'])
		

		for i, scale in enumerate(['front','back']):
			idx = pd.Series(vertical_scale[i] , name = 'h/H')
			setattr(self, scale + '_rho', pd.DataFrame(index = idx, columns = columns) )

		for scale, strip in itertools.product(['front','back'], ['door','box'] ):
			df = getattr(self,scale +'_' + strip + '_strip')
			# horizontal mean of each strip
			getattr(self,scale + '_rho')[self.time, strip, 'mean'].fillna( value = np.mean(df, axis = 1) , inplace = True)
			# horizontal standard deviation of each strip
			getattr(self,scale + '_rho')[self.time, strip, 'std'].fillna( value = np.std(df, axis = 1) , inplace = True)

				
				
				
				
					

	def interface_height(self, vertical_scale, methods = ['threshold','grad','grad2','canny'], **kwargs):
		'''finds the interface height between the ambient and buoyant fluid to compare against prediction
		method = str - threshold - interface is define at a threshold value			
						grad - find the maximum gradient
						grad2 - finds the max turning point and scales it on either the front of the 
						box or the back relative to the camera position and which way parallax is working
						canny - use canny edge deteciton algorithm
						'''
		
		# initialise dataframe with h/H data in place
		columns = pd.MultiIndex.from_product([[self.time], methods], names = ['time','algo_method'])

		for i, scale in enumerate(['front','back']):
			setattr(self, scale + '_interface', pd.DataFrame(index = self.front_box_strip.columns, columns = columns) )

		if 'threshold' in methods:
			try:
				def get_first_non_null(x):
					if x.first_valid_index() is None:
						return 0 
					else:
						return x.first_valid_index()
				# conditional any number greater than 'thres_val' is NaN

				for scale in ['front','back']:
					data = getattr(self, scale + '_box_strip')
					# print(getattr(self,scale + '_interface')[self.time, 'threshold'])
					# exit()
					getattr(self,scale + '_interface')[self.time, 'threshold'].fillna(value = data[data < kwargs['thres_val']].apply(get_first_non_null, axis = 0), inplace = True )

			except KeyError as e:
				print('threshold method requires' + e + 'kwarg')
		
		if 'grad' in methods or 'grad2' in methods:
			try:
				

				def rollingmean(x):
					rolling_mean = x.rolling(kwargs['rolling_mean'], center = True).mean()
					# fill in nan caused by the .rolling (using min_period would create large gradients)
					rolling_mean.fillna(method = 'ffill', inplace = True)
					rolling_mean.fillna(method = 'bfill', inplace = True)
					return rolling_mean

				if 'grad' in methods:

					def max_gradient(x):
						rolling_mean = rollingmean(x)
						return np.argmax(np.abs(np.gradient(rolling_mean)))

					for i, scale in enumerate(['front','back']):
						data = getattr(self, scale + '_box_strip')
						# print(data)
						# print(pd.Series(vertical_scale[i][data.apply(max_gradient, axis = 0)]))
						# print(data.apply(max_gradient, axis = 0))
						# print(getattr(self,scale + '_interface')[self.time, 'grad'])
						# exit()
						getattr(self,scale + '_interface')[self.time, 'grad'].fillna(value = pd.Series(vertical_scale[i][data.apply(max_gradient, axis = 0)], 
						index = self.front_box_strip.columns), inplace = True)

					# filter the interface using the median of the 19 values around it
					self.front_interface[self.time,'grad'] = pd.Series(medfilt(self.front_interface[self.time,'grad'], kernel_size= kwargs['median_filter']),
					 index = self.front_box_strip.columns)
					self.back_interface[self.time,'grad'] = pd.Series(medfilt(self.back_interface[self.time,'grad'], kernel_size= kwargs['median_filter']), 
					index = self.front_box_strip.columns)

					# smooth out with a rolling_mean
					self.front_interface[self.time,'grad'] = self.front_interface[self.time,'grad'].rolling(25, center = True, min_periods=1).mean()
					self.back_interface[self.time,'grad'] = self.back_interface[self.time,'grad'].rolling(25, center = True, min_periods=1).mean()

				else:
					horizontal_average = self.front_box_strip.mean(axis = 1)
					#print(horizontal_average.shape)
					def max_gradient_second_order(x):
						rolling_mean = rollingmean(x)
						return np.argmax(np.gradient(np.gradient(rolling_mean)))

					def min_gradient_second_order(x):
						rolling_mean = rollingmean(x)
						return np.argmin(np.gradient(np.gradient(rolling_mean)))
					
					
					getattr(self,'front_interface')[self.time, 'grad2'].fillna(value = pd.Series(vertical_scale[0][max_gradient_second_order(horizontal_average)]* 
					np.ones(self.front_box_strip.shape[1])), inplace = True)
					getattr(self,'back_interface')[self.time, 'grad2'].fillna(value = pd.Series(vertical_scale[1][min_gradient_second_order(horizontal_average)]* 
					np.ones(self.back_box_strip.shape[1])), inplace = True)


				
			except KeyError as e:
				print('grad method requires' + str(e) + 'kwarg')
			

		if 'canny' in methods:
			pass


# -------------------------------------------------------
#Functions
# -------------------------------------------------------

def get_image_fid(rel_imgs_dir, *img_ext):
    """Function to get a list of file IDs to import.
    rel_imgs_dir = string - relative file path to this script
    *img_ext = string - extensions you want to list the IDs of"""
    try:
        os.chdir(os.path.dirname(__file__)) # Change working directory to the directory of this script
        os.chdir(rel_imgs_dir)
        fid = {}
        for exts in img_ext:
            exts = '*.' + exts.split('.')[-1] # try to ensure if the input is "".txt" or "txt" it doesn't matter
            values = []
            for file in glob.glob(exts):
                # remove the file extension
                values.append(file.split('.')[0])
                values.sort()
            fid[str(exts[1:])] = values
        return fid
    except NameError as e:
        print(e)
    except AttributeError as e:
        print(e)
    except TypeError as e:
        print(e)

def background_img_mean(bg_imgs):
    '''returns a list of np.array of mean rgb channels from the input images'''
    # if the background images need black level offsetting
    if (bg_imgs[0].status['black_level'] == False) and (bg_imgs[0].ext == 'ARW') : 
        bg_imgs = [img.black_offset() for img in bg_imgs]
    
    result = []
    for color in ['red', 'green', 'blue']:
        BG = np.zeros( (getattr(bg_imgs[0], color).shape) ) # empty array
        for img in bg_imgs:
    	    BG += getattr(img, color) # add image channel
        BG = np.ceil( BG / len(bg_imgs) ) # divide by length
        BG[BG == 0] = 1 # Change all values of 0 to one so that they can be dividedby / normalised
        result.append(BG)

    return result # need to return both the mean

def prep_background_imgs(bg_imgs):
    '''Calls the functions above to apply to the list of backgrounnd images''' 
    if not os.path.isfile(bg_imgs[0].img_loc + 'initial_crop_area.csv'): # if csv file doesn't exist
        print('Choose crop for base image')
        crop_pos = bg_imgs[0].choose_crop()
        save_dict(bg_imgs[0].img_loc, crop_pos, csv_name = 'initial_crop_area')
    else:
        crop_pos = read_dict(bg_imgs[0].img_loc, csv_name = 'initial_crop_area')
	
    for img in bg_imgs:
        img.crop_img(crop_pos) #crop images

    bg_mean = background_img_mean(bg_imgs) # find mean of crop.
	

    return (bg_mean, crop_pos)



def save_dict(img_loc, mydict , csv_name ):
	'''function will save crop_pos to .csv file so that we don't have to put it in all the time
	crop_pos: tuple (xy top left corner, width, height)
	purpose: str options:
	 'initial' for cutting down the raw file to the box
	 'analysis' for specifying the area to analyse '''
	with open(img_loc + csv_name + '.csv', 'w') as csv_file:
		for key in mydict.keys():
			if isinstance( mydict[key], np.ndarray ):
				for x in mydict[key]:
					csv_file.write('%s, %f\n'%(key,x))
			else:
				csv_file.write('%s, %s\n'%(key,mydict[key]))


def read_dict(dict_loc, csv_name):
	'''Read in crop data from csv to dictionary'''
	with open(dict_loc + csv_name + '.csv','r') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		d = {}

		if csv_name == 'box_dims':
			h = []
			for row in readCSV:
				k, v = row
				if  k == 'h/H':
					h.append(float(v))
				else:
					d[k] = float(v)
			d['h/H'] = h
		else:
			for row in readCSV:
				k, v = row
				d[k] = int(v)

	return d

def box_dims(img): # cant just apply to the analysis space as you can't see the
	'''pick the dimensions of the box returns the (x,y) coordinates of the top left and bottom 
	right corner of both the front and back of the box as well as the level of the door. IN PIXELS '''
	# Show an image in interactive mode
	plt.ion()
	ax = plt.axes()
	ax.imshow(img.image)

	# get a door level
	response = 'no'
	while 'y' not in response.lower():
		door = int(input('Input the level of the door by eye:'))
		l = Line2D( [0, img.image.shape[1]], [door,door] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l)
		response = input('Are you happy with the door level?  Do you want to continue? [Y/N]')
		l.set_visible(False)
	
	# get dimensions of the back of the box
	response = 'no'
	print('BACK of the box dimensions')
	while 'y' not in response.lower():
		y1_back = int(input('Input the level of the top of the BACK of the box:'))
		l = Line2D( [0, img.image.shape[1]], [y1_back,y1_back] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l)
		y2_back = int(input('Input the level of the bottom of the BACK of the box:'))
		l2 = Line2D( [0, img.image.shape[1]], [y2_back,y2_back] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l2)
		x1_back = int(input('Input the location of the lefthand side of the BACK of the box: '))
		l3 = Line2D([x1_back,x1_back], [0, img.image.shape[0]] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l3)
		x2_back = int(input('Input the location of the righthand side of the BACK of the box: '))
		l4 = Line2D([x2_back,x2_back], [0, img.image.shape[0]] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l4)
		plt.draw()
		response = input('Are you happy with the dimensions given for the back of the box?  Do you want to continue? [Y/N]')
		l.set_visible(False)
		l2.set_visible(False)
		l3.set_visible(False)
		l4.set_visible(False)

	# get dimensions of the front of the box
	response = 'no'
	print('FRONT of the box dimensions')
	while 'y' not in response.lower():
		y1_front = int(input('Input the level of the top of the FRONT of the box:'))
		l = Line2D( [0, img.image.shape[1]], [y1_front,y1_front] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l)
		y2_front = int(input('Input the level of the bottom of the FRONT of the box:'))
		l2 = Line2D( [0, img.image.shape[1]], [y2_front,y2_front] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l2)
		x1_front = int(input('Input the location of the lefthand side of the FRONT of the box: '))
		l3 = Line2D([x1_front,x1_front], [0, img.image.shape[0]] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l3)
		x2_front = int(input('Input the location of the righthand side of the FRONT of the box: '))
		l4 = Line2D([x2_front,x2_front], [0, img.image.shape[0]] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l4)
		plt.draw()
		response = input('Are you happy with the dimensions given for the front of the box?  Do you want to continue? [Y/N]')
		l.set_visible(False)
		l2.set_visible(False)
		l3.set_visible(False)
		l4.set_visible(False)

	plt.ioff()	
	plt.close()
		
	return {'door' : door, 'b_x1' : x1_back, 'b_y1' : y1_back, 'b_x2': x2_back, 'b_y2':y2_back , 
	'f_x1': x1_front,'f_y1':y1_front, 'f_x2': x2_front, 'f_y2':y2_front }

def make_dimensionless(img,box_dims,analysis_area):
	'''converts the pixels of the analysis area into dimensionless form based on the dimensions of the box
	returns dict of door level, vertical and horizontal scale and camera centre for both the front and back of the box'''

	front_vertical_scale = np.linspace(1,0,box_dims['f_y2'] - box_dims['f_y1']) # create scale of front box between 0 and 1
	back_vertical_scale = np.linspace(1,0,box_dims['b_y2'] - box_dims['b_y1']) # create scale of  front box between 0 and 1

	#slicing the scale to just cover the analysis area
	f_v_scale = front_vertical_scale[ int(analysis_area['y1'] - box_dims['f_y1']) : int(analysis_area['y1']+ analysis_area['height'] - box_dims['f_y1'])]
	b_v_scale = back_vertical_scale[ int(analysis_area['y1'] - box_dims['b_y1']) : int(analysis_area['y1']+ analysis_area['height'] - box_dims['b_y1'])]
	
	

	# now create the horizontal scales
	front_horizontal_scale = np.linspace(0,1,box_dims['f_x2'] - box_dims['f_x1']) # create scale of  front box between 0 and 1
	back_horizontal_scale = np.linspace(0,1,box_dims['b_x2'] - box_dims['b_x1']) # create scale of  front box between 0 and 1

	f_h_scale = front_horizontal_scale[ int(analysis_area['x1'] - box_dims['f_x1']) : int(analysis_area['x1']+ analysis_area['width'] - box_dims['f_x1'])]
	b_h_scale = back_horizontal_scale[ int(analysis_area['x1'] - box_dims['b_x1']) : int(analysis_area['x1']+ analysis_area['width'] - box_dims['b_x1'])]


	door_level = pd.DataFrame(index = ['door'], columns= ['front','back'])
	# door height scaled on the front and back
	door_level.loc['door','front'] = front_vertical_scale[int(box_dims['door'] - box_dims['f_y1'])]
	door_level.loc['door','back'] = back_vertical_scale[int( box_dims['door'] - box_dims['b_y1'])] 

	door_level.to_csv(img.img_loc + 'analysis/door_level.csv', sep = ',', index = True)
	
	
	centre_position = pd.DataFrame(index = ['centre'], columns= ['horizontal','vertical'])
	# give the centre of the raw image a value on the horizonatal and vertical scale
	# if the centre is outside the scale range (outside the box) then...
	if int(img.centre[1] - box_dims['f_y1']) > len(front_vertical_scale): 
		centre_position.loc['centre','vertical'] = front_vertical_scale[int(img.centre[1] - box_dims['f_y1'] - len(front_vertical_scale))] - 1
	else:
		centre_position.loc['centre','vertical'] = front_vertical_scale[int(img.centre[1] - box_dims['f_y1'])]

	# if the centre is outside the scale range (outside the box) then...
	if int(img.centre[0] - box_dims['f_x1']) < 0 :
		centre_position.loc['centre','horizontal'] = front_horizontal_scale[int(img.centre[0] - box_dims['f_x1'] + len(front_horizontal_scale))] - 1
	else:
		centre_position.loc['centre','horizontal'] = front_horizontal_scale[int(img.centre[0] - box_dims['f_x1']) ] 

	centre_position.to_csv(img.img_loc + 'analysis/centre_positon.csv', sep = ',', index = True)

	return (centre_position, door_level ,(f_v_scale,b_v_scale), (f_h_scale, b_h_scale))




#########################################################
# Plotting functions
#########################################################

def plot_density_transient(df , door, time, save_loc, steadystate = 500, number_of_plots = 10):
	'''function will save a figure containing 'x' equally spaced (in time)
	plots of the 1D density profile which appears in the dataframe df,
	'''
	
	plt.style.use('seaborn-white')
	fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True, figsize = (12, 9))
	
	# find index closest to the steadystate time
	idx_ss = min(range(len(time)), key=lambda i: abs(time[i]- steadystate))
	# get regular spacing of images in the transient phase
	space = m.floor(len(time[:idx_ss]) / number_of_plots)
	#overwrite with the subset
	time1 = time[:idx_ss:space]

	for t in time1:
		# plot box strip

		ax1.plot(df[ t, 'box', 'mean'] , df.index, label = str(t) + ' sec' )
		ax1.fill_betweenx(df.index, df[ t, 'box', 'mean']  + 2*df[ t, 'box', 'std'], 
		df[ t, 'box', 'mean'] - 2*df[ t, 'box', 'std'], alpha = 0.2)
		ax1.set_xlim( [0, 1] )
		ax1.set_title('Box strip')
		
		ax1.set_ylabel('h/H')
		ax1.set_xlabel('$A$')	
		ax1.legend()

		# plot door	
		ax2.plot(df[ t, 'door', 'mean'] , df.index, label = str(t) + ' sec' )
		ax2.fill_betweenx(df.index, df[ t, 'door', 'mean'] - 2*df[ t, 'door', 'std']  , 
		df[ t, 'door', 'mean'] + 2*df[ t, 'door', 'std'], alpha = 0.2)
		ax2.set_xlim( [0, 1] )
		ax2.set_title('Door strip')
		
		ax2.set_ylabel('h/H')
		ax2.set_xlabel('$A$')	
		ax2.legend()


	ax1.plot([0,1], [door['front'], door['front']], label = 'door_level')
	ax2.plot([0,1], [door['front'], door['front']], label = 'door_level')
	ax1.set_ylim([0, 1])
	fig.suptitle('vertical density profiles' )
	plt.savefig(save_loc + 'rho_profile_transient.png')
	plt.close()

def plot_density(img, door, theory_df, interface):
	'''saves plot of the density profiles scaled on the front of the box
	img - class RAW_img instance
	door - level of the door scale on the front of the box
	theory_df - dataframe with the theory  steadystate interface height'''
	plot_width = 3
	if not os.path.exists(img.img_loc + 'analysis/single_density_profiles'):
		os.mkdir(img.img_loc + 'analysis/single_density_profiles')
	
	plt.style.use('seaborn-white')
	fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (16, 8))

	theory_interface = theory_df.loc[img.bottom_opening_diameter, img.side_opening_height]

	for strip, (axis , scale) in itertools.product(['box', 'door'], zip([ax1, ax3],['front','back'])):
		# Density profiles
		axis.plot(getattr(img, scale + '_rho')[img.time, strip, 'mean'], getattr(img, scale + '_rho').index, label = strip + ' strip' )
		axis.fill_betweenx(getattr(img, scale + '_rho').index, getattr(img, scale + '_rho')[img.time, strip, 'mean']  + 2*getattr(img, scale + '_rho')[ img.time, strip, 'std'], 
		getattr(img, scale + '_rho')[ img.time, strip, 'mean'] - 2*getattr(img, scale + '_rho')[ img.time, strip, 'std'], alpha = 0.2)		
		
		if strip == 'box':
			# interface height
			axis.plot([0,plot_width],[getattr(img, scale + '_interface')[img.time, interface].mean(), getattr(img, scale + '_interface')[img.time, interface].mean()], label = 'interface height', ls = '--')
			axis.fill_between( [0,plot_width], getattr(img, scale + '_interface')[img.time, interface].mean() + 2*getattr(img, scale + '_interface')[img.time, interface].std(), 
			getattr(img, scale + '_interface')[img.time, interface].mean() - 2*getattr(img, scale + '_interface')[img.time, interface].std(), alpha = 0.2)
			axis.set_xlim( [0, plot_width] )
			axis.set_ylim( [0, max(getattr(img, scale + '_rho').index)] )
			axis.set_title(scale.capitalize() + ' - Uncalibrated density profiles')
			axis.set_ylabel('h/H')
			axis.set_xlabel('$A$')
			# door height	
			axis.plot([0,plot_width], [door[scale], door[scale]], label = 'door_level', color = 'r')
			# steady state interface height
			axis.plot([0,plot_width], [theory_interface, theory_interface], label = 'steady state', ls = '--')
			axis.legend()

	#find the closest index to the door in pixels so that it can be plotted on the image
	door_idx =  min( range(img.front_rho[img.time].shape[0]) , key = lambda i : abs(img.front_rho[img.time].index[i]- door.loc['door','front']) )

	#find the closest index of the interface in pixels so that it can be plotted on the image
	# print(img.front_rho[img.time].shape[0])
	# print(img.front_rho[img.time].index)
	# print(img.front_interface[img.time,interface])
	# exit()
	front_interface = img.front_interface[img.time,interface] 
	front_rho = img.front_rho[img.time]
	interface_idx = [ min( range(front_rho.shape[0]), key = lambda i : abs(front_rho.index[i]- x)) for x in front_interface]
	

	ax2.plot([0, len(img.front_door_strip.columns)+ len(img.front_box_strip.columns)], [door_idx, door_idx], label = 'door_level', color = 'r')
	ax2.plot( np.arange(len(img.front_interface)) + img.front_door_strip.shape[1], interface_idx, label  = 'interface height - front_scale', color = 'green' )
	ax2.text( len(img.front_door_strip.columns)/2 , len(img.front_door_strip.index)/2 , 'door strip', color = 'r', rotation = 90)
	ax2.text( len(img.front_door_strip.columns)+len(img.front_box_strip.columns)/2 , len(img.front_box_strip.index)/2 , 'box strip', color = 'r', rotation = 90)
	ax2.plot( [len(img.front_door_strip.columns), len(img.front_door_strip.columns)] , [0, len(img.front_box_strip.index)] , color = 'r' )
	image = ax2.imshow( pd.concat( [img.front_door_strip, img.front_box_strip], axis = 1 ), aspect = 'auto', cmap = 'inferno', vmin = 0, vmax = plot_width)
	plt.colorbar(image,ax = ax2, orientation = 'vertical')
	legend = ax2.legend()
	frame = legend.get_frame()
	frame.set_facecolor('white')

	ax2.set_ylim( [len(img.front_rho.index), 0 ] )
	ax2.set_title('Processed Image')
	ax2.axis('off')
	fig.suptitle( f'Side opening height: {str(img.side_opening_height)}mm \n Bottom opening diameter: {str(img.bottom_opening_diameter)}mm \n {img.filename} - {str(img.time)}sec' )
	plt.savefig(img.img_loc + '/analysis/single_density_profiles/rho_profile_' + str(img.time) + 'secs.png')
	plt.close()
	del(image)



def plot_density_compare_scales(rel_data_dir, data, door, theory_df, exp_conditons):
	'''saves plot of the density profiles
	rel_data_dir - location of the data so it goes in the right folder
	data - this is a list of dataframes to plot
	[0] - density profile scaled on the front of the box
	[1] - density profile scaled on the back of the box
	[2] - interface height scaled on the front of the box
	[3] - interface height scaled on the back of the box
	door_scale - door level
	theory_df - dataframe with the theory  steadystate interface height'''

	time = data[2].name

	if not os.path.exists(rel_data_dir + 'compare_scales'):
		os.mkdir(rel_data_dir + 'compare_scales')
	
	plt.style.use('seaborn-white')
	fig = plt.figure(figsize = (12, 9))
	ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
	
	for i, scale, c in zip(range(2), ['front','back'], ['red','blue']):
		#density profiles
		ax1.plot(data[i]['box', 'mean'], data[i].index , label = 'box strip -' + scale , color = c)
		ax1.fill_betweenx(data[i].index , data[i]['box', 'mean']  + 2*data[i]['box', 'std'], 
		data[i]['box', 'mean'] - 2*data[i]['box', 'std'], alpha = 0.2, color = c)
		#interface height
		ax1.plot([0,1],[data[i+2].mean(), data[i+2].mean()], label = 'interface height - front', ls = '--', color = c)
		ax1.fill_between( [0,1], data[i+2].mean() + 2*data[i+2].std(), data[i+2].mean() - 2*data[i+2].std(), alpha = 0.2, color = c)
		#door level
		ax1.plot([0,1], [door[scale], door[scale]], label = 'door_level - ' + scale , lw = 2, color = c)

	ax1.set_xlim( [0, 1] )
	
	ax1.set_ylim( [0, max(data[0].index.tolist() + data[1].index.tolist())] )
	ax1.set_title('Uncalibrated transmittance profiles')
	ax1.set_ylabel('h/H')
	ax1.set_xlabel('$A$')	

	theory_interface = theory_df.loc[exp_conditons['bod'], exp_conditons['soh']]
	ax1.plot([0,1], [theory_interface, theory_interface], label = 'steady state', ls = ':', lw = 2 , color = 'black')
	ax1.legend()
	
	ax1.set_title( f'Side opening height: {str(exp_conditons["soh"])}mm \n Bottom opening diameter: {str(exp_conditons["bod"])}mm \n {str(time)}sec' )
	plt.savefig(rel_data_dir + 'compare_scales/' + str(time) + 'secs.png')
	plt.close()