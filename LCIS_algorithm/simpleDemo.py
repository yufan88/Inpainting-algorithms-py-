import numpy as np
import matplotlib.pyplot as mplot
import matplotlib.image as mpimg
from lcis import lcis, bandmatrix

if __name__ == '__main__':


    # image path & sampling ratio
    Im = mpimg.imread('lena512.bmp')
    SmpRatio = 0.2

    # creat mask
    mask_Array = np.random.rand(Im.shape[0],Im.shape[1])
    mask_Array = 1*(mask_Array<SmpRatio)

    # sampled image
    print('The sampling ratio is', SmpRatio)
    Im_sampled = np.multiply(mask_Array,Im)
    imgplot = mplot.imshow(Im_sampled)
    mplot.show()

    # reconstruction
    print('Reconstruction starts')
    Im_reconstrcution = lcis (Im_sampled, mask_Array, maxiter = 2500, stepsize = 3)
    imgplot = mplot.imshow(Im_reconstrcution)
    mplot.show()








































