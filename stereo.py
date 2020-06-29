from tools.datamanager import Batch
from tools.datamanager import occupancyGridVisualization
from tools.datamanager import TSDFVisualization
import h5py
import numpy as np
import open3d as o3d
import logging


def jointsVisualization(batch, instance):
    """

    :param batch:
    :param instance:
    :return:
    """
    if len(batch.data3D) == 0:
        logging.error('Import 3D joint data first.\n')
        quit()

    allJointsOnHand = np.array([batch.data3D[batch.jointNames[joint]][instance] for joint in range(21)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(allJointsOnHand)

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    allBonesOnHand = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(allJointsOnHand),
                                          lines=o3d.utility.Vector2iVector(lines))

    o3d.visualization.draw_geometries([pcd, allBonesOnHand, o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])])


def jointsAnimation(batch):
    """

    :param batch:
    :return:
    """
    # Custom visualization object.
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if len(batch.data3D) == 0:
        logging.error('Import 3D joint data first.\n')
        quit()

    allJointsOnHand = np.array([batch.data3D[batch.jointNames[joint]][0] for joint in range(21)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(allJointsOnHand)

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    allBonesOnHand = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(allJointsOnHand),
                                          lines=o3d.utility.Vector2iVector(lines))

    # Initialization.
    vis.add_geometry(pcd)
    vis.add_geometry(allBonesOnHand)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]))

    # Running.
    for instance in range(batch.batchSize):
        allJointsOnHand = np.array([batch.data3D[batch.jointNames[joint]][instance] for joint in range(21)])
        pcd.points = o3d.utility.Vector3dVector(allJointsOnHand)
        allBonesOnHand.points = o3d.utility.Vector3dVector(allJointsOnHand)
        vis.update_geometry(pcd)
        vis.update_geometry(allBonesOnHand)
        vis.poll_events()
        vis.update_renderer()

    # Termination.
    vis.destroy_window()


counting1 = Batch(batchName='B1Counting', commDepth='SK_color_',
                  startImage='0', endImage='1499')
counting1.getDepth('segmentedDepth\\stereo\\B1Counting')
# counting1.getCsv('stereo\\joint_xyz\\B1Counting_SK')

random1 = Batch(batchName='B1Random', commDepth='SK_color_',
                startImage='0', endImage='1499')
random1.getDepth('segmentedDepth\\stereo\\B1Random')
# random1.getCsv('stereo\\joint_xyz\\B1Random_SK')

counting2 = Batch(batchName='B2Counting', commDepth='SK_color_',
                  startImage='0', endImage='1499')
counting2.getDepth('segmentedDepth\\stereo\\B2Counting')
# counting2.getCsv('stereo\\joint_xyz\\B2Counting_SK')

random2 = Batch(batchName='B2Random', commDepth='SK_color_',
                startImage='0', endImage='1499')
random2.getDepth('segmentedDepth\\stereo\\B2Random')
# random2.getCsv('stereo\\joint_xyz\\B2Random_SK')

counting3 = Batch(batchName='B3Counting', commDepth='SK_color_',
                  startImage='0', endImage='1499')
counting3.getDepth('segmentedDepth\\stereo\\B3Counting')
counting3.getCsv('stereo\\joint_xyz\\B3Counting_SK')

random3 = Batch(batchName='B3Random', commDepth='SK_color_',
                startImage='0', endImage='1499')
random3.getDepth('segmentedDepth\\stereo\\B3Random')
random3.getCsv('stereo\\joint_xyz\\B3Random_SK')

counting4 = Batch(batchName='B4Counting', commDepth='SK_color_',
                  startImage='0', endImage='1499')
counting4.getDepth('segmentedDepth\\stereo\\B4Counting')
# counting4.getCsv('stereo\\joint_xyz\\B4Counting_SK')

random4 = Batch(batchName='B4Random', commDepth='SK_color_',
                startImage='0', endImage='1499')
random4.getDepth('segmentedDepth\\stereo\\B4Random')
# random4.getCsv('stereo\\joint_xyz\\B4Random_SK')

counting5 = Batch(batchName='B5Counting', commDepth='SK_color_',
                  startImage='0', endImage='1499')
counting5.getDepth('segmentedDepth\\stereo\\B5Counting')
# counting5.getCsv('stereo\\joint_xyz\\B5Counting_SK')

random5 = Batch(batchName='B5Random', commDepth='SK_depth_',
                startImage='0', endImage='1499')
random5.getDepth('segmentedDepth\\stereo\\B5Random')
# random5.getCsv('stereo\\joint_xyz\\B5Counting_SK')

counting6 = Batch(batchName='B6Counting', commDepth='SK_color_',
                  startImage='0', endImage='1499')
counting6.getDepth('segmentedDepth\\stereo\\B6Counting')
# counting5.getCsv('stereo\\joint_xyz\\B6Counting_SK')

random6 = Batch(batchName='B6Random', commDepth='SK_depth_',
                startImage='0', endImage='1499')
random6.getDepth('segmentedDepth\\stereo\\B6Random')
# random6.getCsv('stereo\\joint_xyz\\B6Random_SK')


# jointsVisualization(counting3, 0)
# jointsAnimation(counting3)

# counting1.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# counting2.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# counting3.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# counting4.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# counting5.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# counting6.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
#
# random1.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# random2.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# random3.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# random4.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# random5.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# random6.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)

# TSDFVisualization(data)
random3.pointCloudVisualization(80)
# og = random3.getOccupancyGrid(900, volumeResolutionValue=32, normalize=False)

# random3.pointCloudAnimation()
