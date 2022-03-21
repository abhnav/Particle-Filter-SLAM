The main loop for particle filiter is in main.py

lidar_map.py contains the MAP specification and utilities to update the map, find correlation, and
calculating and storing the Lidar Raster maps. The update step uses the precomputed rasters from disk.

cam.py contains stereo camera utilities to calculate the depth and the world coordinates. the world
coordinates are also calculated and stored to disk for offline texture (though online texture is also done in main.py)

plot_texture.py contains code to add texture colors. It can also add texture offline when run standalone by
reading the MAP, robot locations, and stereo world coordinates from the disk.

