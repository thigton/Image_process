import glob
import os
import RAW_img
import numpy as np
import pprint
import matplotlib.pyplot as plt

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
        BG /= len(bg_imgs) # divide by length
        result.append(BG)
    return result # need to return both the mean


def crop_background_imgs(bg_imgs):
    '''crops the background images
    bg_imgs should be a list of RAW_img class objects'''
    crop_pos = bg_imgs[0].choose_crop() 
    for img in bg_imgs:
        img.crop_img(crop_pos) #crop images
    return crop_pos
    
    
def prep_background_imgs(bg_imgs):
    '''Calls the functions above to apply to the list of backgrounnd images'''
    crop_pos = crop_background_imgs(bg_imgs)
    bg_mean = background_img_mean(bg_imgs)
    return (bg_mean, crop_pos)




#-------------------------------------------#

if __name__ == '__main__':

    # Chose Parameters
    rel_imgs_dir = './Data/190228_2/' # File path relative to the script
    file_ext = '.JPG' # JPG is working ARW gets error but this might be because I corrupted all the data using git control


    # Get list of file names
    file_ids = get_image_fid(rel_imgs_dir, file_ext)
    filenames = file_ids[file_ext]

    # Get background images
    BG_ids = get_image_fid(rel_imgs_dir + 'BG/', file_ext)
    BG_filenames = BG_ids[file_ext]

    #crop background imgs and get crop_pos  
    (BG, crop_pos) = prep_background_imgs([RAW_img.Raw_img(rel_imgs_dir + 'BG/', f, file_ext) for f in BG_filenames])
    #plt.imshow(BG[0], aspect = 'equal', cmap = 'gray')
    #plt.show()
    
    #----------------------------------------------------------------

    count = 0
    for f in filenames[::-1]: # Analyse in reverse so that the crop images are of the steady state
        # Image preprocessing ========================

         # import image
        img = RAW_img.Raw_img(rel_imgs_dir, f, file_ext)

        # realign

        # undistort

         #crop image
        img.crop_img(crop_pos)

        # black offset
        #img.black_offset()

        #normalise images
        img.normalise(BG) 

        # convert pixels to a real life scale

        # dye calibration

        # Image analysis ============================

        # split into vertical strips to analyse
        if count == 0:
            analysis_area = img.choose_crop()
            strips = RAW_img.define_analysis_strips(img, analysis_area, 400, display=True)
        else:
            strips  = RAW_img.define_analysis_strips(img, analysis_area, 400)

        
        # get 1d density distribution
        density_profiles = RAW_img.one_d_density(strips,10)
        print(density_profiles.head())
        print(density_profiles.shape)
        # get the interface position
        exit()

    #    if count % 10 == 0:
        #    img.save_histogram()

        # save cropped red image
        img.disp_img(disp = False, crop= True, save = True, channel = 'red')

        # housekeeping
        print( str(count+1) + ' of ' + str(len(filenames)) + ' images processed')
        count += 1
        

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