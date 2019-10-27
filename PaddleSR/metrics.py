import numpy as np
import math

__all__ = ['psnr', 'ssim']


# psnr=10*log10((2^n-1)^2/mse)
def psnr(im1, im2, n=8):
    im1 = im1.astype('float32')
    im2 = im2.astype('float32')

    mse = np.mean((im1 - im2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10((pow(2, n) - 1.0) ** 2 / mse)


def ssim(im1, im2, multichannel=False, **kwargs):
    from skimage.measure import compare_ssim
    return compare_ssim(im1, im2, multichannel=multichannel, **kwargs)
