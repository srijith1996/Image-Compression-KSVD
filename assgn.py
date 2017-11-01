import os
import numpy as np
import matplotlib.pyplot as plt

BLK_SIZE = 15
dirname = '../orl_faces/dummy/'

def read_pgm(pgm_file):
    '''
        Read image in a Portable Grayscale Map (pgm) file

        pgm_file - the input pgm file

        Returns numpy array containing the image

    '''

    # check if pgm file
    assert pgm_file.readline() == 'P5\n'
    
    [width, height] = [int(i) for i in pgm_file.readline().split()]
    depth = int(pgm_file.readline())

    assert depth <= 255

    image = []
    for x in xrange(height):

        row = []
        for y in xrange(width):
            row.append(ord(pgm_file.read(1)));
        image.append(row)

    image = np.array(image)

    return image


image_dict = dict()
for im in os.listdir(dirname):

    ex_file = open(dirname + im, 'rb')
    image = read_pgm(ex_file)
    #print image

    image_dict[im] = image

    ht = np.size(image, 0)
    wd = np.size(image, 1)

    #print ht, wd

    ht_pad = BLK_SIZE - (ht % BLK_SIZE)
    pad_top_size = int(np.floor(ht_pad/2))
    pad_bot_size = ht_pad - pad_top_size
    wd_pad = BLK_SIZE - (wd % BLK_SIZE)
    pad_rt_size = int(np.floor(wd_pad/2))
    pad_lt_size = wd_pad - pad_rt_size

    #print pad_top_size, pad_bot_size, pad_rt_size, pad_lt_size
    image = np.lib.pad(image, ((pad_top_size, pad_bot_size), 
        (pad_lt_size, pad_rt_size)), 'constant')

    subim_vects = np.empty([BLK_SIZE * BLK_SIZE, 0])
    
    #print np.shape(image)

    for i in xrange(np.size(image, 0)/BLK_SIZE):
        for j in xrange(np.size(image, 1)/BLK_SIZE):
            
            subimage = image[i*BLK_SIZE:(i+1)*BLK_SIZE, 
                j*BLK_SIZE:(j+1)*BLK_SIZE]

            subimage = np.reshape(subimage, [BLK_SIZE * BLK_SIZE, 1])

            subim_vects = np.append(subim_vects, subimage, axis = 1)

    #assert (np.size(image, 0)/BLK_SIZE * np.size(image, 1)/BLK_SIZE) == np.size(subim_vects, 1)

