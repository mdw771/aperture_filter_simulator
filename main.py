import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.misc import imread, imsave


img = imread('input.jpg').astype('float')
filter = imread('filter.jpg').astype('float')

if filter.ndim == 3:
    filter = filter[:, :, 0]
filter /= filter.max()

filter_eq_size = np.zeros(img.shape[:2])
filter_eq_size[:filter.shape[0], :filter.shape[1]] = filter
filter_fft = fft2(filter_eq_size)

res = filter_eq_size = np.zeros_like(img)
for i in range(res.shape[2]):
    res[:, :, i] = np.abs(ifft2(fft2(img[:, :, i]) * filter_fft))

imsave('output.jpg', res, format='jpeg')