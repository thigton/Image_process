import glob
import os
import RAW_img
import sys



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


rel_imgs_dir = './190305/' # File path relative to the script
file_ids = get_image_fid(rel_imgs_dir, '.ARW')
filenames = file_ids['.ARW']
img = {}
for f in filenames:
    print(f + ' being imported')
    img[f] = RAW_img.Raw_img(rel_imgs_dir, f,save_green=True)
    print('img: ' +str(sys.getsizeof(img)))

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