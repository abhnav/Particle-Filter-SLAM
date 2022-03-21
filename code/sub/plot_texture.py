import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pdb
import pickle
import glob

def load_map_path(pth):
  with open(pth, "rb") as fp:
    loc, loc_timestamps, MAP = pickle.load(fp)
  return loc, loc_timestamps, MAP

def update_color(ptt, color, rt, MAP):
  # color = color[np.logical_and(ptt[:,2] < 0.5, ptt[:,2] > -5)]
  # ptt = ptt[np.logical_and(ptt[:,2] < 0.5, ptt[:,2] > -5)]
  mp = MAP['mcol']
  color = color[ptt[:,2] < 0.5]
  ptt = ptt[ptt[:,2] < 0.5]

  rott = np.array([
    [np.cos(rt[2]), -np.sin(rt[2])],
    [np.sin(rt[2]), np.cos(rt[2])],
    ])

  p_2d = ptt[:,:2] @ rott.T
  p_2d = p_2d + rt[None, :2]

  p_2d[:,0] = np.ceil((p_2d[:,0] - MAP['xmin']) / MAP['res'] ) - 1
  p_2d[:,1] = np.ceil((p_2d[:,1] - MAP['ymin']) / MAP['res'] ) - 1

  ids = p_2d.astype(np.int16)
  mp[ids[:,1], ids[:,0], :] = color

def get_path_map(MAP, loc):
  mp = MAP['map'].copy()
  mp[mp>0] = 1
  mp[mp<0] = -1

  # Get the plot
  lc = np.array(loc)
  lp = lc[:,:,3]
  mx = lp.argmax(axis=1)
  gg = lc[range(len(lp)), mx, :]
  gg[:,0] = np.ceil((gg[:,0] - MAP['xmin'])/MAP['res']) - 1
  gg[:,1] = np.ceil((gg[:,1] - MAP['ymin'])/MAP['res']) - 1
  xc = gg[:,0].astype(int)
  yc = gg[:,1].astype(int)
  mp[yc,xc] = 2 # path
  return mp[::-1,:]

def create_color_map(loc, loc_timestamps, stereo_timestamps,
                     stereo_data, MAP, nochange=False):
  if(nochange):
    return MAP['mcol'][::-1,:,:]
  MAP['mcol'] = np.zeros((MAP['sizey'],MAP['sizex'],3), dtype=np.uint8) 
  i = 0
  j = 0
  print(f"Starting to add texture")
  while(i < len(loc_timestamps) or j < len(stereo_timestamps)):
    vote = np.full(2,1e20)
    if i<len(loc_timestamps):
      vote[0] = loc_timestamps[i]
    if j<len(stereo_timestamps):
      vote[1] = stereo_timestamps[j]
    mi = np.argmin(vote)
    mt = vote[mi]
    if(mi == 0): # location update, no texture
      i = i+1
      continue
    # Add texture otherwise
    best_pi = np.argmax(loc[i][:,3])
    update_color(stereo_data[j][0], stereo_data[j][1],
                 loc[i][best_pi,:3], MAP)
    j = j+1

  return MAP['mcol'][::-1, :, :]

if __name__ == "__main__":
  with open("stereo_depth.dat", "rb") as fp:
    stereo_timestamps, stereo_data = pickle.load(fp)
  stereo_timestamps = np.array(stereo_timestamps)

  Np = 10
  Neff = 2
  pth = f"./results/slam_out_{Np}_{Neff}.dat"
  loc, loc_timestamps, MAP = load_map_path(pth)

  path_map = get_path_map(MAP, loc)
  map_col = create_color_map(loc, loc_timestamps, stereo_timestamps,
                       stereo_data, MAP, nochange=False)
  # map_col = create_color_map(loc, loc_timestamps, stereo_timestamps,
                       # stereo_data, MAP, nochange=True)

  sns.heatmap(path_map)
  plt.show()
  plt.imshow(map_col)
  plt.show()
  pdb.set_trace()


