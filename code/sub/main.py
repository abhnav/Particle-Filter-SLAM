import numpy as np
import time
import pdb
import pickle

from pr2_utils import * 
from cam import *
from lidar_map import *
from plot_texture import *

base = "./"
enc_t, enc_d = read_data_from_csv(base + "data/sensor_data/encoder.csv")
lid_t, lid_d = read_data_from_csv(base + "data/sensor_data/lidar.csv")
fog_t, fog_d = read_data_from_csv(base + "data/sensor_data/fog.csv")

tic = time.time()
with open("raster_map_new.pt", "rb") as fp:
  raster_map = pickle.load(fp)
print(f"time to load lidar raster:{time.time() - tic}")

tic = time.time()
with open("stereo_depth.dat", "rb") as fp:
  stereo_timestamps, stereo_data = pickle.load(fp)
print(f"time to load stereo raster:{time.time() - tic}")
stereo_timestamps = np.array(stereo_timestamps)

# Number of particles
Np = 40
Neff = 10
# Np = 30
# Neff = Np // 5
# Np = 40
# Neff = Np // 4
sts = np.zeros((Np, 4)) # x,y,theta,alpha
sts[:,3] = 1/Np

i = 0 # encoder iterator
j = 0 # yaw iterator
k = 0 # lidar iterator
ll = 0 # stereo iterator
# k = len(lid_t) # to test between fog and encoder

rn = []
rp = []
vel = 0
base_enc = enc_d[0]
base_enc_t = enc_t[0]
ts = min(enc_t[0], fog_t[0], lid_t[0], stereo_timestamps[0]) #initial time 

# location and time
loc = [sts.copy()]
loc_timestamps = [ts]

# control error covariance matrix
cerr = np.array([
      [1e-06, 0, 0], # meters 2*std = 0.002 meter
      [0, 1e-06, 0],
      [0, 0, 2.5000000000000004e-11] # radians 2*std = 0.00001
      ])
np.random.seed(353)

tic = time.time()
while(i<len(enc_t) or j<len(fog_t) or k<len(lid_t) or
        ll<len(stereo_timestamps)):
  tt = i+j+k+ll
  if((tt+1) % 10000 == 0):
    print(f"time epoch:{time.time() - tic}")
    print(f"in loop:{i,j,k}, total:{tt}")
    print(f"average effective count:{np.mean(rn)}, std:{np.std(rn)}")
    print(f"average max prob :{np.mean(rp)}, std:{np.std(rp)}")
    rn = []
    rp = []
    tic = time.time()

  vote = np.full(4,1e20)
  if i<len(enc_t):
    vote[0] = enc_t[i]
  if j<len(fog_t):
    vote[1] = fog_t[j]
  if k<len(lid_t):
    vote[2] = lid_t[k]
  if ll<len(stereo_timestamps):
    vote[3] = stereo_timestamps[ll]

  mi = np.argmin(vote)
  mm = vote[mi]
  if(mi == 0): # predict position
    new_enc = enc_d[i]
    dd = new_enc - base_enc
    dt = mm - base_enc_t
    dis_l = (np.pi/4096) * dd[0] * 0.623479
    if(dis_l == 0):
      vl = 0
    else:
      vl = dis_l / dt

    dis_r = (np.pi/4096) * dd[1] * 0.622806
    if(dis_r == 0):
      vr = 0
    else:
      vr = dis_r / dt

    vel = (vl+vr)/2
    sts[:,0] = sts[:,0] + vel*(mm - ts)*np.cos(sts[:,2])
    sts[:,1] = sts[:,1] + vel*(mm - ts)*np.sin(sts[:,2])
    con_err = np.random.multivariate_normal([0,0,0], cerr, size=Np)
    sts[:,:3] = sts[:,:3] + con_err
    base_enc = new_enc
    base_enc_t = enc_t[i]
    i = i+1

  elif(mi == 1): # predict yaw
    sts[:,0] = sts[:,0] + vel*(mm - ts)*np.cos(sts[:,2])
    sts[:,1] = sts[:,1] + vel*(mm - ts)*np.sin(sts[:,2])
    sts[:,2] = sts[:,2] + fog_d[j][2]
    con_err = np.random.multivariate_normal([0,0,0], cerr, size=Np)
    sts[:,:3] = sts[:,:3] + con_err
    j = j+1

  elif(mi == 3): #add texture
    # update color pivoted on best alpha position
    best_pi = np.argmax(sts[:,3])
    update_color(stereo_data[ll][0], stereo_data[ll][1],
                 sts[best_pi,:3], MAP)
    ll = ll+1

  else: # update state based on lidar
    sts[:,0] = sts[:,0] + vel*(mm - ts)*np.cos(sts[:,2])
    sts[:,1] = sts[:,1] + vel*(mm - ts)*np.sin(sts[:,2])

    p_val, l_val = raster_map[k]
    # check correlation
    cor = []
    for m in range(Np):
      cr = map_cor(sts[m][:3], p_val, l_val, MAP)
      cor.append(cr)
    # update alpha
    # take softmax
    cor = cor - max(cor)
    ss = np.exp(cor).sum()
    prob = np.exp(cor) / ss

    sts[:,3] = sts[:,3] * prob
    if(sts[:,3].sum() == 0):
      print("bad sum")
      pdb.set_trace()

    sts[:,3] = sts[:,3] / sts[:,3].sum()
    # update map based on best alpha index
    best_pi = np.argmax(sts[:,3])
    map_upd(sts[best_pi,:3], p_val, l_val, MAP) # update map from best point
    # resample
    nef = 1 / (sts[:,3] * sts[:,3]).sum()
    rn.append(nef)
    rp.append(sts[best_pi,3])

    if(nef<Neff):
      # resample
      new_p = np.random.choice(np.arange(Np), size=Np, replace=True, p=sts[:,3])
      sts = sts[new_p]
      sts[:,3] = 1/Np
    k = k+1

  ts = mm
  loc.append(sts.copy())
  loc_timestamps.append(ts)


def save_map_path(loc, MAP, loc_timestamps):
  with open(f"./results/slam_out_{Np}_{Neff}.dat", "wb") as fp:
    pickle.dump([loc,loc_timestamps,MAP], fp)

save_map_path(loc, MAP, loc_timestamps)

pdb.set_trace()
