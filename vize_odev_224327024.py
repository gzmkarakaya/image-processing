# BIL552 - Vize Odev - No: 224327024 , Isim: Z.Gizemnur Karakaya
# reset -f
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# 1. Görüntüyü yükleme histogramını çizdirme
img = cv2.imread('goruntu3.png', 0)
plt.figure(figsize=(15,10))
plt.subplots_adjust(hspace=1)
plt.subplot(721),plt.imshow(img,cmap='gray'),plt.title('Original Image')
plt.subplot(722).hist(img.flatten(),256,[0,256]),plt.title('Original Histogram')

# 2. Histogram eşitleme uygulama, histogram eşitleme sonrası görüntüyü ve histogramını çizdirme
equ_img = cv2.equalizeHist(img)
plt.subplot(723),plt.imshow(equ_img, cmap='gray'),plt.title('Equalized Image')
plt.subplot(724).hist(equ_img.flatten(), 256, [0, 256]),plt.title('Equalized Histogram')

# 3. Görüntüye bicubic interpolasyon ile iki kat büyütme, görüntüyü ve histogramını çizdirme
resized_img = cv2.resize(equ_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plt.subplot(725),plt.imshow(resized_img, cmap='gray'),plt.title('Resized Equalized Image (x2)')
plt.subplot(726).hist(resized_img.flatten(), 256, [0, 256]),plt.title('Resized Histogram')

# 4. Görüntüye unsharp masking ve highboost filtering uygulama, görüntüyü ve histogramını çizdirme
gauss = cv2.GaussianBlur(resized_img, (7, 7), 0)
# Apply Unsharp masking
unsharp_img = cv2.addWeighted(resized_img, 2, gauss, -1, 0)
highboost_img = cv2.addWeighted(resized_img,1, unsharp_img, 2 ,0)
plt.subplot(727),plt.imshow(highboost_img, cmap='gray'),plt.title('Highboost Image')
plt.subplot(728).hist(highboost_img.flatten(), 256, [0, 256]),plt.title('Highboost Histogram')

# 5. Highboost sonucunun dft2 sini alma dft2 yi shift etme, görüntüyü ve genlik spektrumunu çizdirme.
size = highboost_img.shape
row = size[0]
col = size[1]
I = cv2.dft(np.float32(highboost_img), flags=cv2.DFT_COMPLEX_OUTPUT)
I_shift = np.fft.fftshift(I)
magnitude_spectrum = 20*np.log(cv2.magnitude(I_shift[:,:,0],I_shift[:,:,1]))

plt.subplot(729),plt.imshow(highboost_img, cmap='gray'),plt.title('Highboost Image')
plt.subplot(7,2,10),plt.imshow(magnitude_spectrum, cmap='gray'),plt.title('Highboost Magnitude Spectrum')

# 6. Görüntüye yarı çapı 200 olan gauss alçak geçiren filtre uygulama gauss filtresini çizdirme
D0 = 200
#Gaussian LPF
H = [[math.exp(-((i-col/2)**2+(j-row/2)**2)/(2*D0**2)) for i in range(col)] for j in range(row)]
plt.subplot(7,2,11),plt.imshow(H, cmap='gray'),plt.title('Gaussian Mask')

# 7. Görüntünün ters dft2 sini alıp görüntünün son halini çizdirme
If = np.zeros((row,col,2))
If[:,:,0]= I_shift[:,:,0]*H
If[:,:,1]= I_shift[:,:,1]*H
f_ishift = np.fft.ifftshift(If)
img_back = cv2.idft(f_ishift)
plt.subplot(7,2,12),plt.imshow(img_back[:,:,0], cmap='gray'),plt.title('Gaussian Filtered Image')

plt.show()

