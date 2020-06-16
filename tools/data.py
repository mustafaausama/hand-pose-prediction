"""
Implementation of class Batch to visualize point cloud and find region of interest.
This is an example file.
"""

from tools.datamanager import Batch
from tools.datamanager import occupancyGridVisualization
from tools.datamanager import getActualDepth
from tools.datamanager import getCoordinatesFromDepth
import cv2
import numpy as np
import open3d as o3d


def segment(batch, instance, normalize=False, volumeResolutionValue=32):
    # Import depth image of the given index.
    depth = cv2.imread(batch.imagesDepth[instance])
    # depth = cv2.medianBlur(depth, 5)

    x_depth = 50
    y_depth = 90
    scale_depth = 256
    roiStart_depth = (y_depth, x_depth)
    roiEnd_depth = (y_depth + scale_depth, x_depth + scale_depth)

    # Merging the two channel depth data into one variable.
    actualDepth = getActualDepth(depth)
    actualDepth = actualDepth[roiStart_depth[0]:roiEnd_depth[0], roiStart_depth[1]:roiEnd_depth[1]]
    print(np.min(actualDepth))

    xyz = getCoordinatesFromDepth(actualDepth, normalize=normalize)

    # Generates the actual Point Cloud.
    pointCloud = o3d.geometry.PointCloud()
    pointCloud.points = o3d.utility.Vector3dVector(xyz)

    # Generates the Axis Aligned Bounding Box.
    axisAlignedBoundingBox = pointCloud.get_axis_aligned_bounding_box()

    # Define the voxel size.
    voxelSize = axisAlignedBoundingBox.get_max_extent() / volumeResolutionValue

    # Define the voxel origin and side lengths.
    origin = axisAlignedBoundingBox.get_center() - (axisAlignedBoundingBox.get_max_extent()) / 2
    totalLength = axisAlignedBoundingBox.get_max_extent()

    # Create the voxel.
    voxelGrid = o3d.geometry.VoxelGrid()
    voxelGrid = voxelGrid.create_dense(origin, voxelSize, totalLength, totalLength, totalLength)

    axisAlignedBoundingBox = voxelGrid.get_axis_aligned_bounding_box()

    o3d.visualization.draw_geometries([pointCloud,
                                       axisAlignedBoundingBox])


# stereo = Batch(batchName='Stereo', commRgb='SK_color_',
#                commDepth='SK_depth_', startImage='0',
#                endImage='1499')
# stereo.getRgb('stereo\\B6Random')
# stereo.getDepth('stereo\\B6Random')
# # stereo.getCsv('stereo\\joint_xyz\\B6Random_SK')


stereo = Batch(batchName='NYU', commRgb='rgb_1_',
               commDepth='depth_1_', startImage='0000001',
               endImage='0000100')
stereo.getRgb('nyu\\dataset_sample\\hand_data')
stereo.getDepth('nyu\\dataset_sample\\hand_data')
# stereo.getCsv('nyu\\joint_xyz\\1')

# stereo.makeVideo()
segment(stereo, 99)
# og = stereo.getOccupancyGrid(99)
# occupancyGridVisualization(og)
# stereo.pointCloudAnimation()
# stereo.roi()
