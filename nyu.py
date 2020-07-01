from tools.datamanager import Batch
from tools.datamanager import TSDFVisualization
import cv2
import numpy as np


def segment(batch):
    synth = Batch(batchName='', commDepth='synthdepth_1_')
    synth.getDepth('nyu\\trainrandom\\synth')
    for instance in range(batch.batchSize):
        synDepth = cv2.imread(synth.imagesDepth[instance])
        depth = cv2.imread(batch.imagesDepth[instance])
        mask = cv2.inRange(synDepth, (0, 1, 0), (256, 256, 256))
        depth = cv2.bitwise_and(depth, depth, mask=mask)

        cv2.imwrite(f'{batch._exportDir}\\{instance}.png', depth)


test = Batch(batchName='test', commDepth='synthdepth_1_')
test.getDepth('nyu\\trainrandom\\synth')
# segment(test)

# segmentedDepth = Batch(batchName='', commDepth='')
# segmentedDepth.getDepth('exports\\test')


TSDFVisualization(test.getAccurateTSDF(33))
# test.makeAccurateTSDF(volumeResolutionValue=32, normalize=False)
# segmentedDepth.pointCloudVisualization(55)
# test.pointCloudVisualization(33)
