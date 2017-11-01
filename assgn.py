import os
import numpy as np
import matplotlib.pyplot as plt

# BLK_SIZE constant for block size in preprocessing
BLK_SIZE = 15
# dirname is the name of the directory containing the images
dirname = '../orl_faces/s1each/'

# read .pgm files
def read_pgm(pgm_file):
    '''
        Read image in a Portable Grayscale Map (pgm) file

        pgm_file - the input pgm file

        Returns numpy array containing the image

    '''

    # check if pgm file
    assert pgm_file.readline() == 'P5\n'
    
    # width and height values are contained in the second header line
    [width, height] = [int(i) for i in pgm_file.readline().split()]

    # depth may or may not be in the third line [ this has to be improved ]
    depth = int(pgm_file.readline())

    # check if the depth value is 8-bit or not
    assert depth <= 255
    
    # image list which will be converted to matrix
    image = []
    for x in xrange(height):

        # each line of the file will have consecutive 8-bit values
        # of pixel intensities
        row = []
        for y in xrange(width):
            row.append(ord(pgm_file.read(1)));
        image.append(row)

    # image converted to numpy array
    image = np.array(image)

    return image


#----------------------------------------------------------
# Script starts here

# store image names and matrices in a dictionary
#image_dict = dict()

# feature vector for all images in the training set
features = np.empty([BLK_SIZE * BLK_SIZE, 0])

# iterate over image names in the directory specified
for im in os.listdir(dirname):

    # open file and read pgm content
    ex_file = open(dirname + im, 'rb')
    image = read_pgm(ex_file)
    #print image

    # append image to the image dictionary : Not necessary for 
    #image_dict[im] = image

    ht = np.size(image, 0)
    wd = np.size(image, 1)

    #print ht, wd


    # compute zero-pads for the block segmentation 
    ht_pad = BLK_SIZE - (ht % BLK_SIZE)
    pad_top_size = int(np.floor(ht_pad/2))
    pad_bot_size = ht_pad - pad_top_size

    # zero-pads for right and left
    wd_pad = BLK_SIZE - (wd % BLK_SIZE)
    pad_rt_size = int(np.floor(wd_pad/2))
    pad_lt_size = wd_pad - pad_rt_size

    # pad the image
    #print pad_top_size, pad_bot_size, pad_rt_size, pad_lt_size
    image = np.lib.pad(image, ((pad_top_size, pad_bot_size), 
        (pad_lt_size, pad_rt_size)), 'constant')

    # subimage vectors in the unrolled columns form
    subim_vects = np.empty([BLK_SIZE * BLK_SIZE, 0])
    
    #print np.shape(image)

    # generate the subim_vects
    for i in xrange(np.size(image, 0)/BLK_SIZE):
        for j in xrange(np.size(image, 1)/BLK_SIZE):
           
            subimage = image[i*BLK_SIZE:(i+1)*BLK_SIZE, 
                j*BLK_SIZE:(j+1)*BLK_SIZE]

            subimage = np.reshape(subimage, [BLK_SIZE * BLK_SIZE, 1])

            subim_vects = np.append(subim_vects, subimage, axis = 1)

    #assert (np.size(image, 0)/BLK_SIZE * np.size(image, 1)/BLK_SIZE) == np.size(subim_vects, 1)
    
    # append features to the final matrix
    features = np.append(features, subim_vects, axis=1)

print features
