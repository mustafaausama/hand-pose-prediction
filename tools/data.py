"""
Implementation of class Batch to visualize point cloud and find region of interest.
This is an example file.
"""

from tools.datamanager import Batch

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
# stereo.pointCloudVisualization(99)
stereo.vocsal(99)
# stereo.pointCloudAnimation()
# stereo.roi()
