import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
from PIL import Image
from os import path
from google.colab.patches import cv2_imshow

def myHM(img_in, img_ref):
  r,g,b = cv2.split(img_in)
  r_ref,g_ref,b_ref = cv2.split(img_ref)

  x = np.zeros(256)
  for i in range(256):
    x[i] = i

  hist_r = np.zeros(256)
  for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        hist_r[r[i][j]] = hist_r[r[i][j]]+1
  hist_g = np.zeros(256)
  for i in range(g.shape[0]):
    for j in range(g.shape[1]):
        hist_g[g[i][j]] = hist_g[g[i][j]]+1
  hist_b = np.zeros(256)
  for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        hist_b[b[i][j]] = hist_b[b[i][j]]+1
  refhist_r = np.zeros(256)
  for i in range(r_ref.shape[0]):
    for j in range(r_ref.shape[1]):
        refhist_r[r_ref[i][j]] = refhist_r[r_ref[i][j]]+1
  refhist_g = np.zeros(256)
  for i in range(g_ref.shape[0]):
    for j in range(g_ref.shape[1]):
        refhist_g[g_ref[i][j]] = refhist_g[g_ref[i][j]]+1
  refhist_b = np.zeros(256)
  for i in range(b_ref.shape[0]):
    for j in range(b_ref.shape[1]):
        refhist_b[b_ref[i][j]] = refhist_b[b_ref[i][j]]+1

  T = np.zeros(256)
  refcdf = np.zeros(257)
  reftot = np.sum(refhist_r)
  refpdf = refhist_r/reftot
  cdf = np.zeros(257)
  tot = np.sum(hist_r)
  pdf = hist_r/tot
  for i in range(1,256):
    cdf[i] = cdf[i-1] + pdf[i]
    refcdf[i] = refcdf[i-1] + refpdf[i]
  for i in range(1,256):
    val = np.full(257, cdf[i])
    T[i] = np.argmin(np.abs(refcdf-val))
  for i in range(img_in.shape[0]):
    for j in range(img_in.shape[1]):
      r[i][j] = T[r[i][j]]

  T = np.zeros(256)
  refcdf = np.zeros(257)
  reftot = np.sum(refhist_g)
  refpdf = refhist_g/reftot
  cdf = np.zeros(257)
  tot = np.sum(hist_g)
  pdf = hist_g/tot
  for i in range(1,256):
    cdf[i] = cdf[i-1] + pdf[i]
    refcdf[i] = refcdf[i-1] + refpdf[i]
  for i in range(1,256):
    val = np.full(257, cdf[i])
    T[i] = np.argmin(np.abs(refcdf-val))
  for i in range(img_in.shape[0]):
    for j in range(img_in.shape[1]):
      g[i][j] = T[g[i][j]]

  T = np.zeros(256)
  refcdf = np.zeros(257)
  reftot = np.sum(refhist_b)
  refpdf = refhist_b/reftot
  cdf = np.zeros(257)
  tot = np.sum(hist_b)
  pdf = hist_b/tot
  for i in range(1,256):
    cdf[i] = cdf[i-1] + pdf[i]
    refcdf[i] = refcdf[i-1] + refpdf[i]
  for i in range(1,256):
    val = np.full(257, cdf[i])
    T[i] = np.argmin(np.abs(refcdf-val))
  for i in range(img_in.shape[0]):
    for j in range(img_in.shape[1]):
      b[i][j] = T[b[i][j]]

  img_out = cv2.merge((r, g, b))
  return img_out

def bounding_box(mask):

	for i in range(1, mask.shape[0]-1):
		for j in range(1, mask.shape[1]-1):
			if(mask[i,j]==0 and mask[i+1,j]==0 and mask[i,j+1]==0 and mask[i+1,j+1]==1):
				top_left = (i+1,j+1)

			if(mask[i,j]==0 and mask[i+1,j]==0 and mask[i,j-1]==0 and mask[i+1,j-1]==1):
				top_right = (i+1,j-1)

			if(mask[i,j]==1 and mask[i+1,j]==0 and mask[i,j-1]==0 and mask[i+1,j-1]==0):
				bottom_left = (i,j)

			if(mask[i,j]==1 and mask[i+1,j]==0 and mask[i,j+1]==0 and mask[i+1,j+1]==0):
				bottom_right = (i,j)
	return np.array([top_left, top_right, bottom_left, bottom_right])
 
def poisson_blending(source, target, mask):
    a = 1
    yr, xr = target.shape[:-1]   
    mask[mask != 0] = 1
    D = scipy.sparse.lil_matrix((xr, xr))
    D.setdiag(-1, -1)
    D.setdiag(4)
    D.setdiag(-1, 1)
    A = scipy.sparse.block_diag([D] * yr).tolil()
    
    A.setdiag(-1, 1*xr)
    A.setdiag(-1, -1*xr)
    laplacian = A.tocsc()
    for y in range(1, yr - 1):
        for x in range(1, xr - 1):
            if mask[y, x] == 0:
                k = x + y * xr
                A[k, k] = 1
                A[k, k + 1] = 0
                A[k, k - 1] = 0
                A[k, k + xr] = 0
                A[k, k - xr] = 0
    A = A.tocsc()

    flatm = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[:, :, channel].flatten()
        target_flat = target[:, :, channel].flatten()            
        b = laplacian.dot(source_flat)*a
        b[flatm==0] = target_flat[flatm==0]
        
        x = spsolve(A, b)
        x = x.reshape((yr, xr))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        target[:, :, channel] = x

    return target

IMG_path = "/content/drive/MyDrive/MIC/lab07_170070021/save2/output_wp/45.png"
ref = np.array(Image.open(IMG_path))
ref_img = cv2.imread(IMG_path)
target = np.array(Image.open("/content/drive/MyDrive/MIC/lab07_170070021/save2/input_images/45.png"))
target_img = cv2.imread("/content/drive/MyDrive/MIC/lab07_170070021/save2/input_images/45.png")
mask = (ref-target)
# box = bounding_box(mask[:,:,0])
# print(box)
mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
cv2_imshow(target_img)
new_mask = myHM(mask, ref_img)
cv2_imshow(new_mask)
mask[(mask[:,:,0]+mask[:,:,1]+mask[:,:,2])>0] = 255
mask_im = Image.fromarray(mask)
result = poisson_blending(ref, target, mask[:,:,0])
result = Image.fromarray(result)
result