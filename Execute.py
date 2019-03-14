import glob
import os
import RAW_img
from collections import defaultdict
import numpy as np
import pprint

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


<<<<<<< HEAD
rel_imgs_dir = './Data/190305/' # File path relative to the script
file_ids = get_image_fid(rel_imgs_dir, '.ARW')

=======
rel_imgs_dir = './190307_2/' # File path relative to the script
file_ids = get_image_fid(rel_imgs_dir, '.JPG')
>>>>>>> d724a36461785a95e0a28c278c32572e8845119d


filenames = file_ids['.JPG']
count = 0
for f in filenames:
<<<<<<< HEAD
    count = 0
    img = RAW_img.Raw_img(rel_imgs_dir, f) # import data
    print(type(img.red))
    #img.black_offset()# black offset
    xy = [500,500]
    width = 1500
    height = 1200
    img.crop_img(xy,width,height,check_crop = True)#crop img
=======
    img = RAW_img.image_process(rel_imgs_dir, f,ext = '.JPG') # import data
    #print(img.raw_image.shape)
    #print(img.raw_image[:,:,0].shape)
    #pprint.pprint(img.metadata)
   # img.black_offset()# black offset
   # Initial crop guesses
    if count == 0 : # Check crop size on the first image
        xy = [3500,300]
        width = 4000
        height = 3000
        img.crop_img(xy,width,height,check_crop = True)#crop img
        xy = img.crop_xy
        width = img.crop_width
        height = img.crop_height
        count += 1
    else:
        img.crop_img(xy, width, height)
        count += 1
>>>>>>> d724a36461785a95e0a28c278c32572e8845119d
    # save
    img.disp_img(disp = False, crop= True, save = True, channel = 'green')



#
#
#for keys, values in img1.metadata.items():
#    print(str(keys) + '  :   ' + str(values))
#
#
#print(img1.metadata['BlackLevel'].split())
#print(type(img1.metadata['BlackLevel'].split()[0]))
#img1.save_histogram()
#xy = [2000,200]
#width = 1500
#height = 1100
#img1.crop_img(xy, width, height, save_red=False)

#plt.imshow(img1.crop_red, aspect = 'equal')
#plt.show()
#plt.axis('off')



#img1.save_histogram(crop=True)
#img1.save_histogram(crop=False)

#try:
#	#img1.disp_histogram()
#	print(img1.size)
#except NameError as e:
#	print(str(e) + ' as an')
#except AttributeError as e:
#	print(e)
#print(dir(img1))

#print(type(img1.raw_image[0,0]))
#print(type(img2[0]))