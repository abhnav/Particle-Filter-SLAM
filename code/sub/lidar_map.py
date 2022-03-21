import numpy as np
import time
import pickle

lidar_t = np.array([0.8349, -0.0126869, 1.76416])
lidar_r = np.array([
    [0.00130201, 0.796097, 0.605167],
    [0.999999, -0.000419027, -0.00160026],
    [-0.00102038, 0.605169, -0.796097]
    ])

# create angles from -5 to 185 in 0.666 increments
ang_deg = np.arange(-5,185.01, 0.666)
ang_rad = (np.pi/180) * ang_deg
ang_x = np.cos(ang_rad)
ang_y = np.sin(ang_rad)

# Map definition and correlation function
MAP = {}
MAP['res']   = 1 #metres
MAP['xmin']  = -100 # metres, extra 100 for lidar scan envelope 
MAP['xmax']  =  1400 # metres, extra 100 for lidar scan envelope 
MAP['ymax']  =  100 # metres, extra 100 for lidar scan envelope 
MAP['ymin']  = -1200 # metres, extra 100 for lidar scan envelope 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizey'],MAP['sizex'])) # log odds will be added, so threshold to prevent overflow
MAP['mcol'] = np.zeros((MAP['sizey'],MAP['sizex'],3), dtype=np.uint8) 

def bres_map(lid): # 286 radial values
  lid[lid==0] = 80
  l_x = lid * ang_x
  l_y = lid * ang_y
  pp = []
  pr = []
  for i in range(len(lid)):
    if(lid[i] == 80): # free till mapped to 80
      pts = bresenham2D(0,0,l_x[i], l_y[i])
      pp.append(pts)
      pr.append(np.full(len(pts[0]), -1))
    elif(lid[i] > 0.5 and lid[i] < 80): # filter invalid measurement
      pts = bresenham2D(0,0,l_x[i], l_y[i])
      pp.append(pts)
      res = np.full(len(pts[0]), -1)
      res[len(pts[0]) - 1] = 1  # the last one is an obstacle
      pr.append(res)

  pp = np.concatenate(pp, axis=1)
  pr = np.concatenate(pr, axis=0)
  return (pp.T,pr)

def map_upd(rob_st, p_car, l_val, MAP):
  mp = MAP['map']
  conf_free = -np.log(4)
  conf_block = np.log(4)
  conf_lim = 6*np.log(4)
  rott = np.array([
    [np.cos(rob_st[2]), -np.sin(rob_st[2])],
    [np.sin(rob_st[2]), np.cos(rob_st[2])],
    ])
  p_t = p_car[:,:2] @ rott.T
  p_t = p_t + rob_st[np.newaxis, :2]
  # convert from meters to cells
  p_t[:,0] = np.ceil((p_t[:,0] - MAP['xmin']) / MAP['res'] ) - 1
  p_t[:,1] = np.ceil((p_t[:,1] - MAP['ymin']) / MAP['res'] ) - 1
  free = p_t[l_val == -1].astype(np.int16)
  block = p_t[l_val == 1].astype(np.int16)
  mp[free[:,1], free[:,0]] += conf_free
  mp[block[:,1], block[:,0]] += conf_block
  # Threshold
  mp[free[:,1], free[:,0]][mp[free[:,1], free[:,0]] < -conf_lim] = -conf_lim
  mp[block[:,1], block[:,0]][mp[block[:,1], block[:,0]] > conf_lim] = conf_lim
  return

def map_cor(rob_st, p_car, l_val, MAP):
  n = 10
  rst = np.zeros((10,len(rob_st)))
  tr = (np.pi/180)*0.666
  rst[:,2] = np.linspace(-tr, tr, num=n) # perturb yaw
  rst = rob_st[np.newaxis, :] + rst

  mp = MAP['map']
  ml = []
  for i in range(n):
    rt = rst[i,:]
    rott = np.array([
      [np.cos(rt[2]), -np.sin(rt[2])],
      [np.sin(rt[2]), np.cos(rt[2])],
      ])
    p_t = p_car[:,:2] @ rott.T
    p_t = p_t + rt[np.newaxis, :2]
    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111)
    # ax.scatter(p_t[:,0], p_t[:,1], c = l_val)
    # plt.show()

    # convert from meters to cells
    p_t[:,0] = np.ceil((p_t[:,0] - MAP['xmin']) / MAP['res'] ) - 1
    p_t[:,1] = np.ceil((p_t[:,1] - MAP['ymin']) / MAP['res'] ) - 1

    lodd = mp[p_t[:,1].astype(np.int16), p_t[:,0].astype(np.int16)]
    lodd[lodd>0] = 1
    # l_val[l_val<0] = 0 # don't match free ones
    lodd[lodd<0] = -1
    olap = lodd.dot(l_val)
    ml.append(olap)
  return max(ml)

if __name__ == "__main__":
  tic = time.time()
  raster_map = []
  for k in range(len(lid_t)):
  # for k in range(10):
    # get observed points
    pp, pl = bres_map(lid_d[k])
    pp = np.c_[pp, np.zeros(len(pp))]
    pp = pp @ lidar_r.T + lidar_t[np.newaxis, :]

    val_ind = np.logical_or(pl == 1, pp[:,2]<10) # either obstacle, or height limit
    p_val = pp[val_ind]
    p_val[:,2] = 0 # take projection for 2d grid
    l_val = pl[val_ind]
    if(k%1000 == 0):
      print(f"{k}:{time.time() - tic}")
      tic = time.time()
    raster_map.append((p_val, l_val))

  with open("raster_map_separate.pt", "wb") as fp:
    pickle.dump(raster_map, fp)


