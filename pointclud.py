from tools.datamanager import Batch

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
# counting3.getCsv('stereo\\joint_xyz\\B3Counting_SK')

random3 = Batch(batchName='B3Random', commDepth='SK_color_',
                startImage='0', endImage='1499')
random3.getDepth('segmentedDepth\\stereo\\B3Random')
# random3.getCsv('stereo\\joint_xyz\\B3Random_SK')

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

counting5 = Batch(batchName='B6Counting', commDepth='SK_color_',
                  startImage='0', endImage='1499')
counting5.getDepth('segmentedDepth\\stereo\\B6Counting')
# counting5.getCsv('stereo\\joint_xyz\\B6Counting_SK')

random6 = Batch(batchName='B6Random', commDepth='SK_depth_',
                startImage='0', endImage='1499')
random6.getDepth('segmentedDepth\\stereo\\B6Random')
# random6.getCsv('stereo\\joint_xyz\\B6Random_SK')

random3.pointCloudVisualization(70)
