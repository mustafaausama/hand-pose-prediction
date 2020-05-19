"""
Contains definition and declaration of 'Batch' class and its method and variables
which can be used to import and manipulate data. An example of this is given in
data.py.
"""

import logging
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


class Batch(object):
    def __init__(self, batchName, commRgb=None, commDepth=None,
                 startImage=None, endImage=None):
        """
        This object contains the color and/or depth images
        as well as the corresponding 3D point data from different
        dataset used in this project.
        This constructor takes five parameters to be initialized

        :param batchName:   A unique name for the to be imported dataset batch.
                            Name it carefully because any processed images will
                            be saved in a directory of this name in exports folder.
        :param commRgb:     A string that is common between all the color images of
                            the dataset. For example in NYU dataset the string
                            'rgb_1_' is repeated between all the color images.
        :param commDepth:   This is similar as previous parameter. It contains
                            the common string used in all the depth images. For NYU
                            it is 'depth_1_'.
        :param startImage:  The starting index of images that the object
                            will import. For example to import NYU dataset from image
                            number 50 to 100 we will set this value to '0000050'.
        :param endImage:    The index of last image that will be imported in
                            this data set. For above given example it will be a string
                            '0000100'.

        :return:            self
        """

        # Public member variables

        self.batchSize = None
        self.imagesRgb = []
        self.imagesDepth = []
        self.joint = None
        self.jointNames = []
        self.data3D = {}

        # Private member variables

        self._startImage = startImage
        self._endImage = endImage
        self._batchName = batchName
        self._commRgb = commRgb
        self._commDepth = commDepth
        self._rgb = False
        self._depth = False
        self._roi = False

        # These directories will be different on different machines.
        # Set them accordingly.
        self._dataDir = 'D:\\DataSets\\'
        self._exportDir = f'D:\\DataSets\\exports\\{batchName}\\'

        # Creates a directory with name of batch.
        Path(self._exportDir).mkdir(parents=True, exist_ok=True)

    def _get(self, dataDir, images, commString):
        # Private method which generates the strings for
        # both rgb and depth images.
        # Should never be called by user.
        for imRange in np.arange(int(self._startImage), int(self._endImage) + 1):
            leading_zeros = '%0' + str(len(self._startImage)) + 'd'
            image_index = leading_zeros % imRange
            images.append(self._dataDir + dataDir + '\\'
                          + commString + image_index
                          + '.png')

    def getRgb(self, dataDir):
        """
        This method generates the strings for rgb images and sets the flags
        self._rgb = True. Note that the strings cannot be overwritten.
        Later, when processing images, these strings with the path of individual
        images will be imported in code using opencv or some other package.
        For example in opencv, to access 'i' image, we can use
        image = cv2.imread(self.imagesRgb[i])

        :param dataDir: The path to rgb image folder relative to parent
                        dataset folder.

        :return:        self.imagesRgb
        """
        if self._commRgb is None:
            return
        if not self._rgb:
            self._get(dataDir, self.imagesRgb, self._commRgb)
            self._rgb = True
            self.batchSize = np.size(self.imagesRgb)
        else:
            self._rgb = False
            self.getRgb(dataDir)
            # Throw a warning if there is already data in
            # the batch.
            logging.warning('Batch already populated.\n'
                            'Data replaced.')

    def getDepth(self, dataDir):
        """
        This method generates the strings for depth images and sets the flags
        self._depth = True. Note that the strings cannot be overwritten.
        Later, when processing images, these strings with the path of individual
        images will be imported in code using opencv or some other package.
        For example in opencv, to access 'i' image, we can use
        image = cv2.imread(self.imagesDepth[i])

        :param dataDir: The path to depth image folder relative to parent
                        dataset folder. Can be same as rgb images.

        :return:        self.imagesDepth
        """
        if self._commDepth is None:
            return
        if not self._depth:
            self._get(dataDir, self.imagesDepth, self._commDepth)
            self._depth = True
            self.batchSize = np.size(self.imagesDepth)
        else:
            self._depth = False
            self.getDepth(dataDir)
            logging.warning('Batch already populated.\n'
                            'Data replaced.')

    def getCsv(self, csvDir):
        """
        This method imports the 3D points data from csv files in the
        given directory. This data is imported and stored in self.data3D
        public member variable. This can take some time for large data.
        This method modifies the data3D member variable.
        self.data3D will become a dictionary with keys as the name of joints
        and each entry at each joint contains a list of size m, where m is the
        number of images in this batch. Each entry in this list is another list
        with size 3. These are the three coordinates of each image.
        The number of joints is stored in self.joint.
        Keep in mind that names of joints are stored in self.jointNames
        so they can be accessed to index through self.data3D.
        For example in NYU data set to access the x, y, z coordinates of 'F1_KNU3_A'
        joint of 'i' image in batch, what we can do is:
        x, y, z = self.data3d['F1_KNU3_A'][i]
        or alternatively the name of joint may not be used.
        x, y, z = self.data3d[self.jointNames[0]][i]
        Make sure that a csv file 'joint_names.csv' is present in the
        directory above the csv directory.

        :param csvDir:  The directory that contains csv files for each
                        point on the hand. This is relative to the parent
                        dataset folder.

        :return:        self.data3D
        """

        # Reads the name of joints
        filenames = np.genfromtxt(self._dataDir + csvDir
                                  + '\\..\\' + 'joint_names.csv',
                                  delimiter=',', dtype='|U15')
        # Garbage control
        # Will delete any empty string left in csv file
        indexToBeDeleted = []
        for i in range(len(filenames)):
            if filenames[i] == '':
                indexToBeDeleted.append(i)
        filenames = np.delete(filenames, indexToBeDeleted)

        # This loop reads the individual joint files.
        for filename in tqdm(filenames, desc=f'CSV {self._batchName}'):
            eachJoint = np.genfromtxt(self._dataDir + csvDir + '\\'
                                      + filename + '.csv', delimiter=',')

            row = []
            # This loop generates a list of size m, in which m is
            # the number of images in batch.
            if len(self._startImage) > 5:
                # For NYU dataset as it starts its images with 1
                for image in np.arange(int(self._startImage) - 1, int(self._endImage)):
                    row.append(eachJoint[image][[1, 2, 3]])
            else:
                # For dataset that start their images with 0
                for image in np.arange(int(self._startImage), int(self._endImage) + 1):
                    row.append(eachJoint[image][[1, 2, 3]])

            # A dictionary is created with keys as the name of joints given in
            # 'joint_names.csv' file.
            self.data3D[filename.replace('.csv', '')] = row
            self.jointNames.append(filename.replace('.csv', ''))
        self.joint = len(self.jointNames)

    def roi(self):
        """
        Finds the region of interest from the images and crops the images
        to 256 x 256 containing the region of interest. Also creates a directory
        of name roi within export directory.

        :return:
        """

        if not self._rgb and not self._depth:
            logging.warning('Batch is empty.\n')
            return

        # Create a directory named roi within batch directory.
        Path(self._exportDir + 'roi').mkdir(parents=True, exist_ok=True)
        self._roi = True

        # Iter through each image and find roi
        for instance in tqdm(range(self.batchSize), desc=f'ROI {self._batchName}'):
            jointsU = []
            jointsV = []

            # iter through each joint and note the U and V coordinate.
            for i in range(self.joint):
                jointsU.append(self.data3D[self.jointNames[i]][instance][0])
                jointsV.append(self.data3D[self.jointNames[i]][instance][1])

            # These value select the roi square region.
            uOffset = 150
            vOffset = 180
            multiplier = 1
            roiU = int(multiplier * max(jointsU)) + uOffset
            roiV = int(multiplier * min(jointsV)) + vOffset

            # Starting and Ending points of roi.
            roiStart = np.array([roiU, roiV])
            roiEnd = np.array([roiU + 256, roiV + 256])

            # Makes sure to add leading zeros to filename.
            leading_zeros = '%0' + str(len(self._startImage)) + 'd'
            imageIndex = leading_zeros % (instance + int(self._startImage))

            # Checks flags to see whether or not desired import had taken place.
            if self._rgb:
                rgb = cv2.imread(self.imagesRgb[instance])

                # Cropping
                roi_rgb = rgb[roiStart[1]:roiEnd[1], roiStart[0]:roiEnd[0]]

                Path(self._exportDir + 'roi\\' + 'rgb').mkdir(parents=True,
                                                              exist_ok=True)
                cv2.imwrite(self._exportDir + 'roi\\rgb\\'
                            + self._commRgb + f'{imageIndex}.png', roi_rgb)
            if self._depth:
                depth = cv2.imread(self.imagesDepth[instance])

                # Cropping
                roi_depth = depth[roiStart[1]:roiEnd[1], roiStart[0]:roiEnd[0]]

                Path(self._exportDir + 'roi\\' + 'depth').mkdir(parents=True,
                                                                exist_ok=True)
                cv2.imwrite(self._exportDir + 'roi\\depth\\'
                            + self._commDepth + f'{imageIndex}.png', roi_depth)

    def pointCloudVisualization(self, instance):
        """
        Takes an index to data and shows the generated point cloud map of the
        data. This point cloud map is generated from depth images so this
        method will not work if depth images are not added to batch.
        Note that this function does not returns as long as the visualization
        window is opened.
        :param instance:    Data index that is going to be visualized.
                            Must be between 0 and self.batchSize-1.

        :return:
        """

        # Checks if depth images are present in batch.
        if not self._depth:
            logging.error('No depth images added.\n')
            return

        # Checks if image index is out of bounds.
        if instance > self.batchSize - 1:
            logging.error(f'Enter the image between 0 and {self.batchSize - 1}.\n')
            return

        # Import depth image of the given index.
        depth = cv2.imread(self.imagesDepth[instance])

        # Checks which of the color channels contain depth information
        # e.g. For NYU dataset it is blue and green.
        if max(depth[:, 0, 0]) == 0:
            hChannel = depth[:, :, 1]
            lChannel = depth[:, :, 2]
        elif max(depth[:, 0, 1]) == 0:
            hChannel = depth[:, :, 0]
            lChannel = depth[:, :, 2]
        elif max(depth[:, 0, 2]) == 0:
            hChannel = depth[:, :, 0]
            lChannel = depth[:, :, 1]
        else:
            logging.error('Depth value is not 10 bits.\n'
                          'Not Supported.\n')
            return

        # Creates the vectors of x and y points.
        xPoints = np.linspace(0, depth.shape[1] - 1,
                              depth.shape[1], dtype=int)
        yPoints = np.linspace(0, depth.shape[0] - 1,
                              depth.shape[0], dtype=int)
        xMesh, yMesh = np.meshgrid(xPoints, yPoints)
        xMesh = np.reshape(xMesh, (1, np.size(xMesh)))
        yMesh = np.reshape(yMesh, (1, np.size(yMesh)))

        # Merging the two channel depth data into one variable.
        hChannel = np.reshape(hChannel, (depth.shape[1], depth.shape[0]))
        lChannel = np.reshape(lChannel, (depth.shape[1], depth.shape[0]))
        lChannel = lChannel + 256
        totalDepth = hChannel + lChannel
        totalDepth = np.reshape(totalDepth, -1)

        # Checks where in image is static depth present.
        depthToBeDeleted = []
        for i in range(len(totalDepth)):
            if totalDepth[i] == totalDepth[0]:
                depthToBeDeleted.append(i)

        # Deletes the static depth value.
        xMesh = np.delete(xMesh, depthToBeDeleted)
        yMesh = np.delete(yMesh, depthToBeDeleted)
        totalDepth = np.delete(totalDepth, depthToBeDeleted)

        # Converts the mesh data to a single (depth.shape[0] x depth.shape[1]) x 3
        # matrix.
        xyz = np.zeros((np.size(xMesh), 3))
        xyz[:, 0] = np.reshape(xMesh, (1, np.size(yMesh)))
        xyz[:, 1] = np.reshape(yMesh, (1, np.size(xMesh)))
        xyz[:, 2] = np.reshape(totalDepth, (1, np.size(totalDepth)))
        maxAll = max([max(xyz[:, 0]), max(xyz[:, 1]), max(xyz[:, 2])])
        minAll = min([min(xyz[:, 0]), min(xyz[:, 1]), min(xyz[:, 2])])

        # Normalize the coordinates between 0 and 1.
        xyz[:, 0] = (xyz[:, 0] - minAll) / (maxAll - minAll)
        xyz[:, 1] = (xyz[:, 1] - minAll) / (maxAll - minAll)
        xyz[:, 2] = (xyz[:, 2] - minAll) / (maxAll - minAll)
        maxAllNormalized = min([max(xyz[:, 0]), max(xyz[:, 1]), max(xyz[:, 2])])
        minAllNormalized = min([min(xyz[:, 0]), min(xyz[:, 1]), min(xyz[:, 2])])

        # cubeStart = 0
        # cubeLen = 1
        # points = [
        #     [cubeStart, cubeStart, cubeStart],
        #     [cubeLen, cubeStart, cubeStart],
        #     [cubeStart, cubeLen, cubeStart],
        #     [cubeLen, cubeLen, cubeStart],
        #     [cubeStart, cubeStart, cubeLen],
        #     [cubeLen, cubeStart, cubeLen],
        #     [cubeStart, cubeLen, cubeLen],
        #     [cubeLen, cubeLen, cubeLen],
        # ]

        cubeStartX = min(xyz[:, 0])
        cubeStartY = min(xyz[:, 1])
        cubeStartZ = min(xyz[:, 2])
        cubeLenX = maxAllNormalized + cubeStartX
        cubeLenY = maxAllNormalized + cubeStartY
        cubeLenZ = maxAllNormalized + cubeStartZ

        points = [
            [cubeStartX, cubeStartY, cubeStartZ],
            [cubeLenX, cubeStartY, cubeStartZ],
            [cubeStartX, cubeLenY, cubeStartZ],
            [cubeLenX, cubeLenY, cubeStartZ],
            [cubeStartX, cubeStartY, cubeLenZ],
            [cubeLenX, cubeStartY, cubeLenZ],
            [cubeStartX, cubeLenY, cubeLenZ],
            [cubeLenX, cubeLenY, cubeLenZ],
        ]
        lines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        colors = [[0, 0, 1] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        # line_set.colors = o3d.utility.Vector3dVector(colors)

        volumeResolutionValue = 32
        voxelSize = max(max(xyz[:, 0]), max(xyz[:, 1]),
                        max(xyz[:, 2])) / volumeResolutionValue
        # print(max(xyz[:, 2]))
        voxelGrid = o3d.geometry.VoxelGrid()

        # Generates the actual Point Cloud.
        # Uncomment the second last line to write the point cloud data.
        pointCloud = o3d.geometry.PointCloud()
        pointCloud.points = o3d.utility.Vector3dVector(xyz)

        axisAlignedBoundingBox = pointCloud.get_axis_aligned_bounding_box()
        center = axisAlignedBoundingBox.get_center()
        print(center)
        center = [[10, 10, 10]]
        center = o3d.utility.Vector3dVector(center)

        voxelGrid.create_from_point_cloud(pointCloud, voxelSize)

        # o3d.io.write_point_cloud(f'{self._batchName}_{instance}.ply', pointCloud)
        o3d.visualization.draw_geometries([pointCloud,
                                           line_set])

    def makeVideo(self, frameRate=60):
        """
        Makes a video file with name of the batch. This is needed for visualization
        of dataset and check the applied algorithm at a glance instead of checking
        each image one by one. By default frame rate is 60fps but can be changed.

        :return:
        """
        # Initializes the variable in which frames will be stored.
        frames = []

        for imageNumber in tqdm(range(self.batchSize),
                                desc=f'Video {self._batchName}'):
            frame = cv2.imread(self.imagesRgb[imageNumber])
            frames.append(frame)

        vidWriter = cv2.VideoWriter(f'{self._batchName}.avi',
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                    frameRate,
                                    (frames[0].shape[1], frames[0].shape[0]))
        for imageNumber in range(len(frames)):
            vidWriter.write(frames[imageNumber])
        # Releases the video so it can be changed outside of this program.
        vidWriter.release()

    def roiSegment(self):
        """
        The content of this function are very case specific and vary wildly from
        image to image. There is not one unified algorithm to segment hands from
        all data sets.
        The following algorithm was used to segment depth images on the basis of
        color images. It is recommended that this function is run on a notebook
        because sliders are added to perfectly tune the hsv values color.

        :return:
        """

        # Tune the roi for color images.
        x_color = 210
        y_color = 40
        scale_color = 340
        roiStart_color = (y_color, x_color)
        roiEnd_color = (y_color + scale_color, x_color + scale_color)

        # Tune the roi for depth images.
        # The reason these are separate because depth images do not
        # map perfectly on color images and need to be aligned.
        x_depth = 231
        y_depth = 90
        scale_depth = 256
        roiStart_depth = (y_depth, x_depth)
        roiEnd_depth = (y_depth + scale_depth, x_depth + scale_depth)

        # Create a directory of name 'seg'.
        Path(self._exportDir + 'seg').mkdir(parents=True,
                                            exist_ok=True)
        for imageNumber in tqdm(range(1500), desc=f'Seg {self._batchName}'):
            # imageNumber = 907 #@param {type:"slider", min:0, max:1499, step:1}
            color = cv2.imread(self.imagesRgb)
            depth = cv2.imread(self.imagesDepth)

            color = color[roiStart_color[0]:roiEnd_color[0],
                    roiStart_color[1]:roiEnd_color[1]]

            hl = 0  # @param {type:"slider", min:0, max:255, step:1}
            hh = 225  # @param {type:"slider", min:0, max:255, step:1}
            sl = 42  # @param {type:"slider", min:0, max:255, step:1}
            sh = 255  # @param {type:"slider", min:0, max:255, step:1}
            vl = 0  # @param {type:"slider", min:0, max:255, step:1}
            vh = 255  # @param {type:"slider", min:0, max:255, step:1}
            color = cv2.resize(color, (256, 256))
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, (hl, sl, vl), (hh, sh, vh))

            mask = cv2.medianBlur(mask, 7)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

            color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            depth = depth[roiStart_depth[0]:roiEnd_depth[0],
                    roiStart_depth[1]:roiEnd_depth[1]]

            # Overlays the color images on top of depth images for visualization
            # combined = np.zeros((256, 256, 3), dtype=np.uint8)
            # combined[:, :, 0] = color[:, :, 0]
            # combined[:, :, 2] = 2*depth[:, :, 2]

            color = cv2.bitwise_and(color, color, mask=mask)

            depth = cv2.bitwise_and(depth, depth, mask=mask)
            depth = cv2.medianBlur(depth, 5)

            # Makes sure to add leading zeros to filename.
            leading_zeros = '%0' + str(len(self._startImage)) + 'd'
            imageIndex = leading_zeros % (imageNumber + int(self._startImage))

            cv2.imwrite(self._exportDir + 'roi\\'
                        + self._commDepth + f'{imageIndex}.png', depth)
