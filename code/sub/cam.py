import numpy as np
import os
import time
import pdb
import pickle
import glob

from pr2_utils import * 

stereo_r = np.array([
    [-0.00680499, -0.0153215, 0.99985],
    [-0.999977, 0.000334627, -0.00680066],
    [-0.000230383, -0.999883, -0.0153234]
    ])
stereo_t = np.array([1.64239, 0.247401, 1.58411])

cx = 6.0850726281690004e+02 # must be in pixels
fx = 8.1690378992770002e+02 # must be in pixels
cy = 2.6347599764440002e+02 # must be in pixels
fy = 8.1156803828490001e+02 # must be in pixels
bs = 475.143600050775 # mm

def compute_stereo_world_points(path_l, path_r):
  image_l = cv2.imread(path_l, 0)
  image_r = cv2.imread(path_r, 0)

  image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
  image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

  image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
  image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

  # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
  stereo = cv2.StereoBM_create(numDisparities=48, blockSize=9) 
  disparity = stereo.compute(image_l_gray, image_r_gray)

  xr = np.arange(disparity.shape[1])
  yr = np.arange(disparity.shape[0])
  xx,yy = np.meshgrid(xr,yr)

  color = image_l[disparity>0]
  # negative disparity indicate no depth information
  zz = np.full(disparity.shape, fx*bs)
  zz[disparity>0] = zz[disparity>0] / disparity[disparity>0]
  zz[disparity<=0] = 0

  xx = xx - cx
  xx = (xx * zz)/fx

  yy = yy - cy
  yy = (yy * zz)/fy

  pts = np.stack([xx[disparity>0],yy[disparity>0],zz[disparity>0]], axis=1)
  ptt = pts @ stereo_r.T
  ptt = ptt/1000
  ptt = ptt + stereo_t[None, :]
  return ptt, color


if __name__ == "__main__":
  # Get world coordinates using depth and save to disk for texture map
  base = "./"
  left_images = glob.glob(base+"../stereo_left/*")
  left_images.sort()
  right_images = glob.glob(base+"../stereo_right/*")
  right_images.sort()
  sync_num = 1113 # only the first 1113 images are in sync

  stereo_data = []
  stereo_timestamps = []
  for img_index in range(sync_num):
    print(f"index:{img_index}")
    ts = os.path.splitext(os.path.basename(left_images[img_index]))[0]
    stereo_timestamps.append(int(ts))
    dat = compute_stereo_world_points(left_images[img_index],
        right_images[img_index])
    stereo_data.append(dat)

  with open("stereo_depth.dat", "wb") as fp:
    pickle.dump([stereo_timestamps, stereo_data], fp)

# limage_names = [os.path.basename(pp) for pp in left_images]
# rimage_names = [os.path.basename(pp) for pp in right_images]
# limage_names.sort()
# rimage_names.sort()
# for img_name in rimage_names:
  # if(ii in limage_names):
    # print(f"Yay:{ii}")
    # cc = cc+1
  # else:
    # print(f"NO:{ii}")

