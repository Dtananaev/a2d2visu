import numpy as np
import glob
import os
from tqdm import tqdm
from PIL import Image
import json
import argparse
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from transforms_a2d2 import transform_from_to, get_transform_from_global, get_transform_to_global, get_axes_of_a_view
import copy
import cv2
def read_json(json_path):
    with open (json_path, 'r') as f:
        data = json.load(f)
        return data


def visualize_lidar(lidar, color=None, figure=None):
    """
    Draw lidar points
    Args:
        lidar: numpy array (n,3) of XYZ
        figure: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """

    if figure is None:
        figure = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )

    # draw origin
    mlab.points3d(
        0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2, figure=figure
    )
    # draw axis
    mlab.plot3d(
        [0, 2], [0, 0], [0, 0], color=(1, 0, 0), tube_radius=None, figure=figure
    )
    mlab.plot3d(
        [0, 0], [0, 2], [0, 0], color=(0, 1, 0), tube_radius=None, figure=figure
    )
    mlab.plot3d(
        [0, 0], [0, 0], [0, 2], color=(0, 0, 1), tube_radius=None, figure=figure
    )

    if color is None:
        s = lidar[:, 2]
    else:
        s = np.arange(len(lidar))
    p3d = mlab.points3d(
        lidar[:, 0],
        lidar[:, 1],
        lidar[:, 2],
        s,
        mode="point",
        scale_factor=3.0,
        colormap="gnuplot",
        figure=figure,
    )
    if color is not None:
        p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(s)
        p3d.module_manager.scalar_lut_manager.lut.table = color
    return figure



def project_lidar_from_to(lidar, src_view, target_view):
    lidar = dict(lidar)
    trans = transform_from_to(src_view, target_view)
    points = lidar['points']
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T 
    lidar['points'] = points_trans[:,0:3]
    
    return lidar


def get_lidar_in_camera_coords(lidar_data, intrinsics):
    y_coords = (lidar_data['row'] + 0.5).astype(np.int)
    x_coords = (lidar_data['col'] + 0.5).astype(np.int)
    depth = lidar_data['depth']  
    hom_coords = np.concatenate((y_coords[:, None], x_coords[:, None], np.ones_like(x_coords)[:, None]), axis=-1)
    norm_coords = (np.matmul(np.linalg.inv(intrinsics), hom_coords.T)).T
    norm_coords = norm_coords / norm_coords[:, -1][:, None]
    point_cloud = norm_coords * depth[:, None]
    return point_cloud

def apply_transform(lidar, transform):
    ones = np.ones_like(lidar[:, 0])
    hom_lidar = np.concatenate( (lidar, ones[:, None]), axis=-1)
    points_trans = (np.dot(transform, hom_lidar.T)).T 

    return points_trans[:, :3]



def get_origin_of_a_view(view):
    return view['origin']


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)
    
    # get origin 
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)
    
    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis
    
    # origin
    transform_to_global[0:3, 3] = origin
    
    return transform_to_global


def get_transform_from_global(view):
   # get transform to global
   transform_to_global = get_transform_to_global(view)
   trans = np.eye(4)
   rot = np.transpose(transform_to_global[0:3, 0:3])
   trans[0:3, 0:3] = rot
   trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])
    
   return trans



def undistort_image(image, cam_name, config):
    if cam_name in ['front_left', 'front_center', \
                    'front_right', 'side_left', \
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
                  np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']
        
        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                      D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, \
                      distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image




def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def map_lidar_points_onto_image(image_orig, pixels_coords, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)
    
    # get rows and cols
    rows = (pixels_coords[:, 1]).astype(np.int)
    cols = (pixels_coords[:, 0]).astype(np.int)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']  
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                        np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
                (1. - pixel_opacity) * \
                np.multiply(image[pixel_rows, pixel_cols, :], \
                colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)

def visualize_dataset(dataset_dir):
    sequences = glob.glob(os.path.join(dataset_dir, "*"))
    config = read_json(os.path.join(dataset_dir, "cams_lidars.json"))

    for seq in sequences:
        if not os.path.isdir(seq):
            continue

        images_list = sorted(glob.glob(os.path.join(seq, "camera", "*", "*.png")))
        for image_path in images_list:
            sensor_name = image_path.split("/")[-2][len("cam_"):]
            intrinsics =  config['cameras'][sensor_name]["CamMatrix"]



            lidar_path = image_path.replace("camera", "lidar").replace(".png", ".npz")
            semseg_path = image_path.replace("camera", "label")

            # load data
            image = np.asarray(Image.open(image_path))
            undist_image = undistort_image(image, sensor_name, config)
            semseg = np.asarray(Image.open(semseg_path))
            lidar_data = np.load(lidar_path)
            lidar = lidar_data["points"]
            # lidar_camera = copy.deepcopy(lidar)
            # lidar_camera[:, 0] = -lidar[:, 1] # x = -y
            # lidar_camera[:, 1] = -lidar[:, 2] # y = -z
            # lidar_camera[:, 2] = lidar[:, 0] # z = x
            
            # pixels_coords = np.matmul(intrinsics, lidar_camera.T).T
            # pixels_coords = pixels_coords[:, :2] / pixels_coords[:, 2][:, None]
            # print(f"wtf")
            # undist_image = map_lidar_points_onto_image(undist_image, pixels_coords, lidar_data, pixel_size=3, pixel_opacity=1)
            # print(f"wtf1")
            # img = Image.fromarray(undist_image.astype("uint8"))
            # img.save("test.png")

            # input()
           

            camera_view = config['cameras'][sensor_name]['view']
            camera_view2 = config['cameras']["side_right"]['view']

            vehicle_view = config['vehicle']['view']
            # view_lidar =   config['lidars']["front_right"]["view"]
            # trans =    get_transform_to_global(view_lidar)
            # trans = transform_from_to(camera_view, view_lidar)
            trans_2 = transform_from_to(camera_view, camera_view2)
            lidar = apply_transform(lidar, trans_2)

            #lidar = apply_transform(lidar, np.linalg.inv(trans))


            figure = visualize_lidar(lidar)
            mlab.show(1)
            input()
            mlab.close(figure)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visu a2d2 transforms")
    parser.add_argument("--dataset_dir", default="/media/denis/F/a2d2/a2d2_semseg_boxes")
    args = parser.parse_args()
    visualize_dataset(args.dataset_dir)




