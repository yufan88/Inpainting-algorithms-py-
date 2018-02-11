# Calculate LCIS result for undersampled images
# YUFAN LUO 2018/1/15


import numpy as np
from numpy import empty,arange,exp,real,imag,pi
from numpy.fft import rfft,irfft


def dct(y):
    N = len(y)
    y2 = empty(2*N,float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = rfft(y2)
    phi = exp(-1j*pi*arange(N)/(2*N))
    return real(phi*c[:N])


def idct(a):
    N = len(a)
    c = empty(N+1,complex)

    phi = exp(1j*pi*arange(N)/(2*N))
    c[:N] = phi*a
    c[N] = 0.0
    return irfft(c)[:N]

def dct2(y):
    M = y.shape[0]
    N = y.shape[1]
    a = empty([M,N],float)
    b = empty([M,N],float)

    for i in range(M):
        a[i,:] = dct(y[i,:])
    for j in range(N):
        b[:,j] = dct(a[:,j])
    return b


def idct2(b):
    M = b.shape[0]
    N = b.shape[1]
    a = empty([M,N],float)
    y = empty([M,N],float)

    for i in range(M):
        a[i,:] = idct(b[i,:])
    for j in range(N):
        y[:,j] = idct(a[:,j])
    return y



def bandmatrix (n):
    band_matrix = np.zeros((n, n))
    for i in range(n-2):
        band_matrix[i + 1, i + 1] = -2
        band_matrix[i + 1, i] = 1
        band_matrix[i + 1, i + 2] = 1
    band_matrix[0, 0] = -1
    band_matrix[0, 1] = 1
    band_matrix[n - 1, n - 1] = -1
    band_matrix[n - 1, n - 2] = 1
    return band_matrix

def lcis (Im_sampled, mask, maxiter = 30000, stepsize = 2, errtol = 2e-4):

    c1 = 1.5
    c2 = 360
    lmda = 350
    n = Im_sampled.shape[0]
    m = Im_sampled.shape[1]

    Al = bandmatrix(n)
    Ar = bandmatrix(m)

    temp = dct2(Al)
    eigenvalues1 = np.zeros((n,1))
    for i in range(n):
        eigenvalues1[i] = temp[i,i]

    temp = dct2(Ar)
    eigenvalues2 = np.zeros((m,1))
    for i in range(m):
        eigenvalues1[i] = temp[i,i]

    deltasquare = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            deltasquare[i,j] = (eigenvalues1[i]+eigenvalues2[j])**2


    d = np.ones((n,m)) + np.ones((n,m))*stepsize*c2+deltasquare*stepsize*c1

    iter = 0

    Im_reconstrucion = Im_sampled

    while iter < maxiter:
        temp = np.zeros((n,m))
        hU0 = dct2(Im_reconstrucion)

        tempatan = np.arctan((Im_reconstrucion*Ar+Al*Im_reconstrucion)/0.3)
        tempatan = -Al*tempatan-tempatan*Ar
        tempatan = tempatan*stepsize

        templambda = np.multiply(mask,(Im_sampled-Im_reconstrucion))*stepsize*lmda

        t = hU0 + dct2(tempatan+templambda) + np.multiply(stepsize*c1*deltasquare,hU0) + c2*stepsize*hU0

        temp = np.divide(t,d)

        tempname = idct2(temp)

        if np.amax(np.amax(np.absolute(Im_reconstrucion-tempname))) < errtol:
            Im_reconstrucion = tempname
            break

        iter += 1

        if not iter%100:
            print('iter', iter)

    return Im_reconstrucion