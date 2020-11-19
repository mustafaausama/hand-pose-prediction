import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tensorflow.keras.models import load_model
from tqdm import tqdm
from gpuinfo import GPUInfo


# data generator to fetch the h5 tsdf file arugemt recieves file no
class Generator():
    def __init__(self):
        import numpy as np
        import pandas as pd
        import h5py

    def get_data(self, file_no):
        import h5py
        import numpy as np

        filename = 'D:/DataSets/nyu/test/TSDF/' + str(file_no) + '.h5'
        # getting 3d input from h5 files
        h5 = h5py.File(filename, 'r')
        input = np.array(h5['TSDF'])
        input = np.reshape(input, (1, 1, 32, 32, 32))
        # VSTOXX futures data
        h5.close()
        inputs = np.array(input).tolist()
        inputs = torch.FloatTensor(inputs)

        return inputs


# while loading the model saved in training we must run this class
# this is needed always as the class was used during trainnig
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 48, (5, 5, 5), padding=1)
        self.conv1_1 = nn.Conv3d(1, 48, (3, 3, 3), padding=0)

        self.pool = nn.MaxPool3d(2, stride=2)

        self.conv2 = nn.Conv3d(96, 96, (5, 5, 5), padding=1)
        self.conv2_2 = nn.Conv3d(96, 96, (3, 3, 3), padding=0)

        self.pool1 = nn.MaxPool3d(2, stride=2)

        self.conv3 = nn.Conv3d(192, 192, (5, 5, 5), padding=1)
        self.conv3_1 = nn.Conv3d(192, 192, (3, 3, 3), padding=0)

        self.pool2 = nn.MaxPool3d(2, stride=2)

        self.ln1 = nn.Linear(in_features=3072, out_features=4096)
        self.ln2 = nn.Linear(in_features=4096, out_features=1024)
        self.out1 = nn.Linear(in_features=1024, out_features=30)

    def forward(self, x):
        # branch one
        x_1 = self.conv1(x)
        x_2 = self.conv1_1(x)

        x_3 = torch.cat([x_2, x_1], dim=1)

        x_4_4 = self.pool(x_3)

        x_4_5 = self.conv2(x_4_4)
        x_4_1 = self.conv2_2(x_4_4)

        x_4 = torch.cat([x_4_1, x_4_5], dim=1)

        x_4 = F.relu(x_4)
        x_4 = self.pool1(x_4)

        x_4_6 = self.conv3(x_4)
        x_4_7 = self.conv3_1(x_4)

        x_4 = torch.cat([x_4_6, x_4_7], dim=1)

        x_4 = F.relu(x_4)

        x_4 = self.pool2(x_4)
        x_4 = x_4.view(-1, 3072)

        x_4 = self.ln1(x_4)
        x_4 = F.relu(x_4)
        x_4 = self.ln2(x_4)
        x_4 = F.relu(x_4)
        x_4 = self.out1(x_4)
        ret = x_4.view(-1, 30)

        return ret



Net = torch.load('C:\\Users\\Use\\Downloads\\pca_30_points_relational_model_2 (1).pt', map_location=torch.device('cuda'))
model = load_model('C:\\Users\\Use\\Downloads\\architecture_3.h5')

Net.eval()
Net.cuda()


gen = Generator()

x = np.genfromtxt('D:\\DataSets\\ground_truth\\test\\joint_x.csv', delimiter=',')
y = np.genfromtxt('D:\\DataSets\\ground_truth\\test\\joint_y.csv', delimiter=',')
z = np.genfromtxt('D:\\DataSets\\ground_truth\\test\\joint_z.csv', delimiter=',')

x = x[:, [0, 3, 6, 9, 12, 15, 18, 21, 24, 26, 28, 30, 31, 32]]
y = y[:, [0, 3, 6, 9, 12, 15, 18, 21, 24, 26, 28, 30, 31, 32]]
z = z[:, [0, 3, 6, 9, 12, 15, 18, 21, 24, 26, 28, 30, 31, 32]]


point_names = np.array(['P1', 'P2', 'R1', 'R2', 'M1', 'M2', 'I1', 'I2', 'T1', 'T2', 'T3', 'W1', 'W2', 'C'])
# mask = np.zeros((5000, 5000, 3), np.uint8)
# for point in range(14):
#     cv2.circle(mask, (int(1500 * (x[1, point]))+1000, int(-1500 * (y[1, point]))+2000), 10, [255, 255, 255], -1)
#     cv2.putText(mask, f'{point}', (int(1500 * (x[1, point]))+1010, int(-1500 * (y[1, point]))+2010), cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 1)
#     cv2.putText(mask, point_names[point], (int(1500 * (x[1, point]))+970, int(-1500 * (y[1, point]))+1980), cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 1)

# cv2.imwrite('mask.png', mask)



log = open("benchmark_time.txt", "w")
logg = open("benchmark_error.txt", "w")


log_time = np.array([])


rng = 8252

errors = np.zeros((rng, 14))

log.write('Using device cuda:0\n')
log.write(torch.cuda.get_device_name(0))
log.write('\nMemory total:    4.0 GB\n')
log.write(f'Memory allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB\n')
log.write(f'Memory cached:    {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB\n')

log.write("\nTime in seconds for each iteration.\n")
logg.write("Percentage error for each point is.\n")

for instance in tqdm(range(rng)):
    inputs = gen.get_data(instance)
    inputs = inputs.cuda()
    
    t = time.time()
    outputs = Net(inputs)
    k = outputs.cpu().detach().numpy()
    predicted_xyz = model.predict(k)
    elapsed_time = time.time() - t
    elapsed_time = elapsed_time*1000

    log.write(f'{instance:04}: {round(elapsed_time, 2):05} ms\n')
    logg.write(f'{instance:04}: ')

    log_time = np.append(log_time, elapsed_time)

    predicted_xyz = predicted_xyz.reshape((14, 3))

    xx = x[instance+1]
    yy = y[instance+1]
    zz = z[instance+1]

    xyz = np.zeros((14, 3))
    xyz[:, 0] = xx
    xyz[:, 1] = yy
    xyz[:, 2] = zz

    error = np.array([])
    for point in range(14):
        logg.write(f'{round(np.linalg.norm(xyz[point] - predicted_xyz[point])*100, 4):07}, ')
        errors[instance, point] = np.linalg.norm(xyz[point] - predicted_xyz[point])*100
    avg = sum(errors[instance, :])/14
    logg.write(f'Best: {round(np.min(errors[instance, :]), 4):07} %, Worst: {round(np.max(errors[instance, :]), 4):07} %, Average: {round(sum(errors[instance, :])/14, 4):07} %\n')


logg.write(f'\nThe mean percentage errors for each point individually are.\n')
logg.write(f'Points : P1     , P2     , R1     , R2     , M1     , M2     , I1     , I2     , T1     , T2     , T3     , W1     , W2     , C\n')
logg.write(f'Mean   : ')

error_mean = np.zeros((14))
for point in range(14):
    error_mean[point] = sum(errors[:, point])/rng
    logg.write(f'{round(error_mean[point], 4):07}, ')

logg.write(f'Average: {round(sum(error_mean)/14, 4):07}\n')

time_average = sum(log_time)/rng

log.write(f'\nThe average time for {rng} instances is {round(time_average, 2)} ms.')

logg.write(f'\nThe Best and Wrost errors for {rng} instances are:\n')
logg.write(f'Best Overall:  {round(np.min(errors), 4)} %\n')
logg.write(f'Worst Overall: {round(np.max(errors), 4)} %')

log.close()
logg.close()

