import glob
import os
import rawpy # RAW file processor - wrapper for libraw / dcraw
import numpy as np
import math as m
import pandas as pd
import statistics as stats
import exiftool
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D      
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
				if row[9] == self.img_loc.split('/')[-2]:
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



	def define_analysis_strips(self, crop_pos, number_of_strips = 1, channel = 'red', display = False):
		'''defines an area of processed image of channel ... to analyse.
		returns a dictionary of strip section and the np.array sitting in value
		img = RAW_img class object
		crop_pos = dictionary from choose_crop() total area to analyse
		strip_width = int in pixels'''
		width = crop_pos['width']
		height = crop_pos['height']
		strip_width = m.floor(width / number_of_strips) # find maximum number of stips based on spacing and width
		x1 = crop_pos['x1']
		y1 = crop_pos['y1']
		# x,y bottom right corner
		x2 = x1 + number_of_strips*strip_width 
		y2 = y1 + height # y-coordinate
		strip_interfaces = [int(i) for i in np.linspace(x1, x2, number_of_strips+1)]
		self.strip_label = np.arange(number_of_strips) # counter for the strips
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
				plt.text( i + round(strip_width/2) , stats.mean([y1,y2]) , str(self.strip_label[j]), color = 'r' )
				plt.draw()
				j += 1
			plt.ioff()
			if not os.path.exists(self.img_loc + 'analysis/'):
				os.makedirs(self.img_loc + 'analysis/')
			plt.savefig(self.img_loc + 'analysis/' + channel + '_channel_analysis_strips.png')
			plt.close()
		# change this to a list of data frames
		self.analysis_space = pd.DataFrame( getattr(self, channel)[y1:y2, x1:x2] )
		self.strips = [pd.DataFrame( getattr(self,channel)[ y1:y2 , strip_interfaces[i] : strip_interfaces[i+1] ] ) for i in self.strip_label]
		self.number_of_strips = number_of_strips

	def one_d_density(self, box_dims, n = 10, save_fig = False):
		'''takes in a dictionary containing np.arrays (strips),
		produces plot, or horizontally average values
		smoothness = number of pixels to do a moving average '''
		

		self.rho = pd.DataFrame(columns = self.strip_label)
		if save_fig == True:
			plt.style.use('seaborn-white')
			fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
			

		for df, l in zip(self.strips, self.strip_label):
			# horizontal mean of each strip
			self.rho[str(l)] = pd.Series(np.mean(df, axis = 1))
			# smoothed out noise with moving average
			self.rho[str(l) + '_' + str(n)]= self.rho[str(l)].rolling( n, min_periods = 1, center = True ).mean()	
			
			if save_fig == True:
				yax = ( np.arange(len(self.rho[str(l)])) - box_dims['top'] )/ ( box_dims['bottom'] - box_dims['top']) # define the yaxis scale
				ax1.plot(self.rho[str(l)] , yax, label = str(l) )
				print(str(yax.shape) + ' and ' + str(self.rho[str(l)].shape) )
		if self.number_of_strips == 1: # if we take the whole analysis area as one, calculate the standard deviation
			self.rho[str(l) + '_std']  = pd.Series(np.std(df, axis = 1))
			
			if save_fig == True:
				ax1.fill_betweenx(yax, self.rho[str(l)] + 2*self.rho[str(l) + '_std'], 
				self.rho[str(l)] - 2*self.rho[str(l) + '_std'], alpha = 0.3, facecolor = 'b', label = '95 %% confidence interval ')

		if save_fig == True:	
			ax1.set_xlim( [0, 1] )
			ax1.set_title('Relative light intensity')
			ax1.plot([0,1], [box_dims['door'], box_dims['door']], label = 'door_level')
			ax1.set_ylabel('h/H')
			ax1.set_xlabel('$I/I_0$')
			ax1.legend()
			processed_image = ax2.imshow(self.analysis_space, aspect = 'auto', cmap = 'plasma')
			processed_image.set_clim(vmin = 0, vmax = 1)
			fig.colorbar(processed_image, ax = ax2 , orientation = 'vertical')
			ax2.set_title('Processed Image')
			fig.suptitle(self.filename + ' - ' + str(self.time) + 'sec' )

			# Save a figure
			plt.savefig(self.img_loc + 'analysis/rho_profile_' + self.filename + '.png')
			plt.close()




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
        save_crop(bg_imgs[0], crop_pos, purpose = 'initial')
    else:
            crop_pos = read_crop(bg_imgs[0], purpose = 'initial')

    for img in bg_imgs:
        img.crop_img(crop_pos) #crop images

    bg_mean = background_img_mean(bg_imgs) # find mean of crop.


    return (bg_mean, crop_pos)



def save_crop(img, crop_pos, purpose = 'initial'):
	'''function will save crop_pos to .csv file so that we don't have to put it in all the time
	crop_pos: tuple (xy top left corner, width, height)
	purpose: str options:
	 'initial' for cutting down the raw file to the box
	 'analysis' for specifying the area to analyse '''
	with open(img.img_loc + purpose + '_crop_area.csv', 'w') as csv_file:
		for key in crop_pos.keys():
			csv_file.write('%s, %s\n'%(key,crop_pos[key]))


def read_crop(img, purpose = 'initial'):
	'''Read in crop data from csv to dictionary'''
	with open(img.img_loc + purpose + '_crop_area.csv','r') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		d = {}
		for row in readCSV:
			k, v = row
			d[k] = int(v)
	return d

def box_dims(img, crop_pos): # cant just apply to the analysis space as you can't see the
	'''returns pixel height of the door level
	CAREFUL - pixel posiiton will be taken after the initial crop and then transformed into the analysis form'''
	# Show an image in interactive mode
	plt.ion()
	ax = plt.axes()
	ax.imshow(img.image)
	response = 'no'
	while 'y' not in response.lower():
		door = int(input('Input the level of the door by eye:'))
		top = int(input('Input the level of the top of the front of the box:'))
		bottom = int(input('Input the level of the bottom of the front of the box:'))
		l = Line2D( [0, img.image.shape[1]], [door,door] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l)
		l2 = Line2D( [0, img.image.shape[1]], [top,top] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l2)
		l3 = Line2D( [0, img.image.shape[1]], [bottom,bottom] , linewidth = 1, color = 'r', visible = True)
		ax.add_line(l3)
		plt.draw()
		response = input('Are you happy with the door level?  Do you want to continue? [Y/N]')
		l.set_visible(False)
		l2.set_visible(False)
		l3.set_visible(False)

	plt.ioff()	
	plt.close()

	return {'door' : door - crop_pos['y1'] , 'top' : top - crop_pos['y1'] , 'bottom' : bottom - crop_pos['y1']}


	
