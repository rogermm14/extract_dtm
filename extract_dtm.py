import numpy as np
import os
import scipy
import skimage
from PIL import Image
import CSF

def interpolate_image(_im_np, method='cubic'):

    # interpolates the input image _im_np
    # method can be 'nearest', 'linear', 'cubic'

    im_np = _im_np.copy()
    undefined_areas_np = np.isnan(im_np)
    defined_areas_np = np.logical_not(undefined_areas_np)
    defined_areas_np = defined_areas_np ^ scipy.ndimage.binary_erosion(defined_areas_np)

    (y, x) = np.where(defined_areas_np)
    z = im_np[y, x]

    if (method != 'nearest'):
        hull = scipy.spatial.ConvexHull(np.array([y, x]).transpose())
        mask_np = np.ones(im_np.shape, dtype=bool)

        rr, cc = skimage.draw.polygon(y[hull.vertices], x[hull.vertices])
        mask_np[rr, cc] = False
        nearest_np = interpolate_image(im_np, method='nearest')

        im_np[mask_np] = nearest_np[mask_np]

        undefined_areas_np = np.isnan(im_np)
        defined_areas_np = np.logical_not(undefined_areas_np)
        defined_areas_np = defined_areas_np ^ scipy.ndimage.binary_erosion(defined_areas_np)

        (y, x) = np.where(defined_areas_np)
        z = im_np[y, x]

    X = np.zeros((z.shape[0], 2))
    X[:, 0] = x
    X[:, 1] = y

    (to_y, to_x) = np.where(undefined_areas_np)
    method_i = 0
    to_z = scipy.interpolate.griddata(X, z, (to_x, to_y), method=method)
    im_np[to_y, to_x] = to_z

    if (np.max(np.isnan(im_np)) == 1):
        print('fail')
        im_np = interpolate_image(im_np, method='nearest')
    return im_np 


def project_cloud_into_utm_grid(xyz, bb, definition, mode):

    # this function projects a point cloud of 3D points in utm coordinates
    # into a utm geographic grid delimited by a rectangular bounding box

    # xyz        : numpy array of shape is (N,3), contains N 3D points in utm coordinates
    #              first column is X, second column Y, third column Z 
    # bb         : numpy array of shape (4,), contains the limits, in utm coordinates, of 
    #              the utm geographic grid where the point cloud xyz will be projected
    #              e.g. bbx = np.array([east_min, east_max, north_min, north_max])
    # mode       : operation employed to project xyz, can be 'min', 'max', 'avg', 'med'
    #              e.g. if 'min' is chosen, the height assigned to each cell will be the
    #                   minimum of the heights of the 3D points that project into that cell
    # definition : resolution of the output geographic grid (in m)
    #              e.g. 0.5
    
    origin = np.array([bb[0], bb[2]])
    w, h = bb[1] - bb[0], bb[3] - bb[2]
    map_w = int(round(w / definition)) + 1
    map_h = int(round(h / definition)) + 1

    map_np = np.zeros((map_h, map_w), dtype=float)
    map_np[:,:] = np.nan
    coords = np.round((xyz[:,:2] - origin) / definition).astype(int)

    if mode == 'min' or mode == 'max':
        if mode == 'min':
            idx = np.flip(np.argsort(xyz[:,2]))
        else:
            idx = np.argsort(xyz[:,2])   
        coords, data_np = coords[idx], xyz[idx]
        dsm_z, dsm_coords = data_np[:,2], coords
    else:
        from itertools import groupby    
        coords_unique, coords_indices = np.unique(coords, return_inverse=True, axis=0)
        sorted_id_z = sorted(list(zip(coords_indices, xyz[:,2])), key=lambda x: x[0])
        groups_id_z = groupby(sorted_id_z, lambda x: x[0])
        dsm_z = []
        if mode == 'avg':
            dsm_z = [np.mean(np.array(list(g))[:,1]) for k, g in groups_id_z]
        else:
            dsm_z = [np.median(np.array(list(g))[:,1]) for k, g in groups_id_z]
        dsm_coords, dsm_z = coords_unique, np.array(dsm_z)

    indices_inside_grid = np.logical_and(dsm_coords[:,1] < map_h, dsm_coords[:,0] < map_w)
    dsm_coords, dsm_z = dsm_coords[indices_inside_grid, :], dsm_z[indices_inside_grid]
    map_np[dsm_coords[:,1], dsm_coords[:,0]] = dsm_z

    if (np.sum(np.logical_not(np.isnan(map_np))) < 3):
        print ('There are less than 3 points.')
    
    raw_map_np = map_np.copy()
    int_map_np = interpolate_image(map_np)
    filt_map_np = scipy.signal.medfilt(int_map_np)
    error_map_np = np.abs(int_map_np - filt_map_np)
    map_np[error_map_np > 3] = np.nan
    int_map_np = interpolate_image(map_np)
    
    return int_map_np, raw_map_np

def main(xyz_fname, bbox_fname, output_dir, bin_dir, definition, save_tmp_files):

    os.makedirs(output_dir, exist_ok=True)
    definition, save_tmp_files = float(definition), bool(int(save_tmp_files))
    file_id = os.path.basename(os.path.splitext(xyz_fname)[0])
    cloud = np.loadtxt(xyz_fname)
    bb = np.loadtxt(bbox_fname)

    # extract ground via cloth simulation (source: https://github.com/jianboqi/CSF)
    # Zhang et al. 2016 "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation"
    csf = CSF.CSF()
    csf.setPointCloud(cloud)
	           
    ground = CSF.VecInt()     # list that will contain the indices of those 3D points classified as ground
    non_ground = CSF.VecInt() # list that will contain the indices of those 3D points classified as non_ground

    # set parameters
    csf.params.bSloopSmooth = False
    csf.params.time_step = 0.45
    csf.params.class_threshold = 0.5
    csf.params.cloth_resolution = 1.0
    csf.params.rigidness = 100
    csf.params.iterations = 500

    # do ground/non_ground classification 
    # 'exportCloth' saves a txt called 'cloth_nodes.txt' with the ground estimation as given by cloth simulation
    csf.do_filtering(ground, non_ground, exportCloth=True)

    # save ground tif to tmp directory
    _, dtm_nan = project_cloud_into_utm_grid(cloud[np.array(ground)], bb, definition, 'min')
    im = Image.fromarray(dtm_nan) 
    ground_fname = os.path.join(output_dir, file_id + '_ground.tif')
    im.save(ground_fname)

    # save cloth to tmp directory
    _, dtm_nan = project_cloud_into_utm_grid(np.loadtxt("cloth_nodes.txt"), bb, definition, 'min')
    im = Image.fromarray(dtm_nan)
    cloth_dtm_fname = os.path.join(output_dir, file_id + '_cloth.tif')
    im.save(cloth_dtm_fname)
    os.system('rm cloth_nodes.txt') # remove tmp file created by CSF

    # post-process dtm to fill missing areas
    fnames, n_steps, cont = [ground_fname, cloth_dtm_fname], 5, 1
    cloth_id = os.path.basename(os.path.splitext(cloth_dtm_fname)[0])
    fnames.extend([os.path.join(output_dir, cloth_id +'_{}.tif'.format(i)) for i in np.arange(1, n_steps+1)])

    # (1) poisson interpolation 
    # to fill NaN values - preserve the contours of gaps and fill unknown areas with a smooth interpolation
    os.system(bin_dir + '/simpois -i {0} -o {1}'.format(fnames[cont], fnames[cont+1]))
    cont += 1

    # (2) downsample 
    # to speed up the process
    os.system(bin_dir + '/downsa i 2 {0} {1}'.format(fnames[cont], fnames[cont+1]))
    cont += 1

    # (3) opening 
    # to remove possible parts of buildings that were not correctly interpreted as ground by CSF
    os.system(bin_dir + '/morsi disk30 opening {0} {1}'.format(fnames[cont], fnames[cont+1]))
    cont += 1
    
    # (4) closing 
    # to further fill cavities caused by the folds of the cloth at bulding footprints
    os.system(bin_dir + '/morsi disk30 closing {0} {1}'.format(fnames[cont], fnames[cont+1]))
    cont += 1
    
    # (5) upsample 
    # to speed up the process (2 is the zoom factor and 3 is the zoom type)
    os.system(bin_dir + '/upsa 2 3 {0} {1}'.format(fnames[cont], fnames[cont+1]))

    os.system('cp ' + fnames[cont+1] + ' ' + os.path.join(output_dir, file_id + '_dtm.tif'))

    if not save_tmp_files:
        for i in range(len(fnames)):
            os.system('rm  ' + fnames[i])

    # EXTRA: save dsm
    dsm_int, _ = project_cloud_into_utm_grid(cloud, bb, definition, 'avg')
    im = Image.fromarray(dsm_int) 
    im.save(os.path.join(output_dir, file_id + '_dsm.tif'))


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
