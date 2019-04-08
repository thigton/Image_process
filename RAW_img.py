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
			x1 = int(input('x-coordinate of the top left corner: '))
			y1 = int(input('y-coordinate of the top left corner: '))
			width = int(input('x-coordinate of the bottom right corner: ')) - x1
			height = int(input('y-coordinate of the bottom right corner: ')) - y1

			# display crop on image
			rect = patches.Rectangle( (x1, y1), width, height, linewidth = 1, edgecolor='r', facecolor = 'none')
			ax.add_patch(rect)
			plt.draw()
			response = input('Are you happy with the crop area?  Do you want to continue? [Y/N]')
		
		# End interactive mode and close figure
		plt.ioff()
		plt.close()
		# return to crop information as dictionary
		return {'x1': x1, 'y1' : y1,  'width' : width, 'height' : height}


	def crop_img(self, crop_pos):
		"""Crops the image to the space you want, if check_crop = True, the image will be displayed 
		and you have the option of re aligning if you want """
		
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



	def define_analysis_strips(self, crop_pos, box_dims, door_strip_width = 100, channel = 'red', save = False):
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

		self.door_strip = pd.DataFrame( getattr(self, channel)[y1:y2, x1:x2], index = pd.Series(box_dims['h/H'] , name = 'h/H'))
		self.box_strip = pd.DataFrame( getattr(self, channel)[y1:y2, x2:x1+width] , index = pd.Series(box_dims['h/H'] , name = 'h/H'))
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

		
		

	def one_d_density(self, box_dims, save_fig = False):
		'''finds horizontal average and standard deviation of box_strip and door_strip) 
		and appends dataframe to a csv file containing this information for all images in the experiment.'''
		
		# initialise dataframe with h/H data in place
		columns = pd.MultiIndex.from_product([[self.time] , ['door', 'box'], ['mean','std'] ], names = ['time','data','attribute'])
		idx = pd.Series(box_dims['h/H'] , name = 'h/H')
		self.rho = pd.DataFrame(index = idx , columns = columns)

		
		for df, l in zip([self.door_strip,self.box_strip], ['door','box']):
			# horizontal mean of each strip
			self.rho[self.time, l, 'mean'].fillna( value = np.mean(df, axis = 1) , inplace = True)
			# horizontal standard deviation of each strip
			self.rho[self.time, l, 'std'].fillna( value = np.std(df, axis = 1) , inplace = True)
					
				


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

def box_dims(img, crop_pos): # cant just apply to the analysis space as you can't see the
	'''CAREFUL - pixel posiiton will be taken after the initial crop and then transformed into the analysis form
	returns a dictionary of the door height relative to the top of the analysis area, and
	a scale of h/H of the analysis area '''
	# Show an image in interactive mode
	plt.ion()
	ax = plt.axes()
	ax.imshow(img.image)
	response = 'no'
	while 'y' not in response.lower():
		door = int(input('Input the level of the door by eye:'))
		l = Line2D( [0, img.image.shape[1]], [door,door] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l)
		top = int(input('Input the level of the top of the back of the box:'))
		l2 = Line2D( [0, img.image.shape[1]], [top,top] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l2)
		bottom = int(input('Input the level of the bottom of the back of the box:'))
		l3 = Line2D( [0, img.image.shape[1]], [bottom,bottom] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l3)
		plt.draw()
		response = input('Are you happy with the door level?  Do you want to continue? [Y/N]')
		l.set_visible(False)
		l2.set_visible(False)
		l3.set_visible(False)

	plt.ioff()	
	plt.close()
	
	scale = np.linspace(1,0,bottom - top) # create scale of box between 0 and 1
	y1_idx = crop_pos['y1'] - top # this should return the index of the top of the analysis area on scale.
	y2_idx = crop_pos['y1']+ crop_pos['height'] - top # same for the bottom
	return {'door' : scale[door-top], 'h/H': scale[y1_idx:y2_idx]}





#########################################################
# Plotting functions
#########################################################

def plot_density_transient(df , box_dims, time, save_loc, steadystate = 500, number_of_plots = 10):
	'''function will save a figure containing 'x' equally spaced (in time)
	plots of the 1D density profile which appears in the dataframe df,
	'''
	
	plt.style.use('seaborn-white')
	fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True, figsize = (12, 9))
	
	# define the y axis
	yax = df.index.tolist()
	# find index closest to the steadystate time
	idx_ss = min(range(len(time)), key=lambda i: abs(time[i]- steadystate))
	# get regular spacing of images in the transient phase
	space = m.floor(len(time[:idx_ss]) / number_of_plots)
	#overwrite with the subset
	time1 = time[:idx_ss:space]

	for t in time1:
		# plot box strip

		ax1.plot(df[ str(t), 'box', 'mean'] , yax, label = str(t) + ' sec' )
		ax1.fill_betweenx(yax, df[ str(t), 'box', 'mean']  + 2*df[ str(t), 'box', 'std'], 
		df[ str(t), 'box', 'mean'] - 2*df[ str(t), 'box', 'std'], alpha = 0.2)
		ax1.set_xlim( [0, 1] )
		ax1.set_title('Box strip')
		
		ax1.set_ylabel('h/H')
		ax1.set_xlabel('$I/I_0$')	
		ax1.legend()

		# plot door strip
		ax2.plot(df[ str(t), 'door', 'mean'] , yax, label = str(t) + ' sec' )
		ax2.fill_betweenx(yax, df[ str(t), 'door', 'mean'] - 2*df[ str(t), 'door', 'std']  , 
		df[ str(t), 'door', 'mean'] + 2*df[ str(t), 'door', 'std'], alpha = 0.2)
		ax2.set_xlim( [0, 1] )
		ax2.set_title('Door strip')
		
		ax2.set_ylabel('h/H')
		ax2.set_xlabel('$I/I_0$')	
		ax2.legend()


	ax1.plot([0,1], [box_dims['door'], box_dims['door']], label = 'door_level')
	ax2.plot([0,1], [box_dims['door'], box_dims['door']], label = 'door_level')
	ax1.set_ylim([0, 1])
	fig.suptitle('vertical density profiles' )
	plt.savefig(save_loc + 'rho_profile_transient.png')
	plt.close()


def plot_density(img, box_dims):
	'''saves plot of the density profiles'''

	if not os.path.exists(img.img_loc + '/analysis/single_density_profiles'):
		os.mkdir(img.img_loc + '/analysis/single_density_profiles')
	
	plt.style.use('seaborn-white')
	fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (12, 9))
	for l in ['box', 'door']:
		ax1.plot(img.rho[img.time, l, 'mean'], box_dims['h/H'], label = l + ' strip' )
		ax1.fill_betweenx(box_dims['h/H'], img.rho[img.time, l, 'mean']  + 2*img.rho[ img.time, l, 'std'], 
		img.rho[ img.time, l, 'mean'] - 2*img.rho[ img.time, l, 'std'], alpha = 0.2)

	ax1.set_xlim( [0, 1] )
	ax1.set_ylim( [0, max(box_dims['h/H'])] )
	ax1.set_title('Uncalibrated density profiles')
	ax1.set_ylabel('h/H')
	ax1.set_xlabel('$I/I_0$')	
	ax1.plot([0,1], [box_dims['door'], box_dims['door']], label = 'door_level', color = 'r')
	ax1.legend()
	
	# find the closest index to the door in pxels so thta it can be plotted on the image
	door_idx =  min(range(len(box_dims['h/H'])), key=lambda i: abs(box_dims['h/H'][i]- box_dims['door']))
	ax2.plot([0, len(img.door_strip.columns)+ len(img.box_strip.columns)], [door_idx, door_idx], label = 'door_level', color = 'r')
	plt.text( len(img.door_strip.columns)/2 , len(img.door_strip.index)/2 , 'door strip', color = 'k', rotation = 90)
	plt.text( len(img.door_strip.columns)+len(img.box_strip.columns)/2 , len(img.box_strip.index)/2 , 'box strip', color = 'k', rotation = 90)
	ax2.plot( [len(img.door_strip.columns), len(img.door_strip.columns)] , [0, len(img.box_strip.index)] , color = 'k' )
	image = ax2.imshow( pd.concat( [img.door_strip, img.box_strip], axis = 1 ), aspect = 'auto', cmap = 'plasma')
	image.set_clim(vmin = 0, vmax = 1)
	ax2.legend()
	fig.colorbar(image, ax = ax2 , orientation = 'vertical')
	ax2.set_title('Processed Image')
	ax2.axis('off')
	fig.suptitle( f'Side opening height: {str(img.side_opening_height)}mm \n Bottom opening diameter: {str(img.bottom_opening_diameter)}mm \n {img.filename} - {str(img.time)}sec' )
	plt.savefig(img.img_loc + '/analysis/single_density_profiles/rho_profile_' + str(img.time) + 'secs.png')
	plt.close()