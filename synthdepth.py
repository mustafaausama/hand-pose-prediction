from tools.datamanager import Batch
from tools.datamanager import TSDFVisualization

synDepth = Batch(batchName='SyntheticDepth', commDepth='synthdepth_1_',
                 startImage='0000173', endImage='0000326')
synDepth.getDepth('synth depth')

# synDepth.pointCloudVisualization(110)
TSDFVisualization(synDepth.getAccurateTSDF(110))
