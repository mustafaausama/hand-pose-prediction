import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import h5py
import numpy as np    
import numpy as np
import pandas as pd
import h5py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):

    def __init__(self,num_unit1,num_unit2,drop_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1,48,(5,5,5), padding=1)
        self.conv1_1 = nn.Conv3d(1,48,(3,3,3), padding=0)
        
        self.pool   =  nn.MaxPool3d(2, stride=2)
        
        self.conv2    =    nn.Conv3d(96,96,(5,5,5), padding=1)
        self.conv2_2  =  nn.Conv3d(96,96,(3,3,3), padding=0)        
        
        self.pool1  =  nn.MaxPool3d(2, stride=2)
        
        self.conv3  =  nn.Conv3d(192,192,(5,5,5), padding=1)
        self.conv3_1  =  nn.Conv3d(192,192,(3,3,3), padding=0)
        
        self.pool2  =  nn.MaxPool3d(2, stride=2)        
        
        # default num_unit1 = 4096
        self.ln1    =  nn.Linear(in_features=3072, out_features=num_unit1)
        self.ln2    =  nn.Linear(in_features=num_unit1, out_features=num_unit2)
        # default num_unit1 = 1024        
        # introuce drop out here 
        self.dropout= nn.Dropout(p=drop_out)
        self.out1   = nn.Linear(in_features=num_unit2, out_features=30)

        
    
    def forward(self, x):
        #branch one
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
        x_4 = self.dropout(x_4)       
        x_4 = self.out1(x_4)
        ret = x_4.view(-1, 30)
             
        return ret
   

def num_flat_features(x):
    x = x
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features    

def get_data_batch(file_no):
    gen = Generator()
    inputs, labels = gen.get_data(file_no)
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])
    file_no = file_no + 1
    inputs_stack, labels_stack = gen.get_data(file_no)
    inputs = torch.cat([inputs, inputs_stack])
    labels = torch.cat([labels, labels_stack])  		     
    return inputs,labels

class Generator():
    def __init__(self):
      #  pass

        self.NYU_X = pd.read_csv('30_POINT_RELATION_FILE.csv')

    def get_data(self,file_no):
                 

        filename = 'TSDF/TSDF/'+str(file_no)+'.h5'
           # getting 3d input from h5 files
        h5 = h5py.File(filename,'r')
        input = np.array(h5['TSDF'])
        input = np.reshape(input,(1,1,32,32,32))
                # VSTOXX futures data
        h5.close()           
        inputs = np.array(input).tolist()
        inputs = torch.FloatTensor(inputs)

        output1 = self.NYU_X.iloc[file_no].values

        output1 = output1[0:30]
        output1 = np.asarray(output1)
        output  = torch.from_numpy(output1).float()
        output  = torch.reshape(output, (1, 30))

        return inputs,output




def trainloop(model,learning_rate,optimizer):
    import torch.optim as optim
    if optimizer == "SGD":
        optimizer = optim.SGD(list(model.parameters()), lr=learning_rate, momentum=0.9)
    if optimizer == "Adam":    
        optimizer = optim.Adam(list(model.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if optimizer == "Adadelta":
        optimizer = optim.Adadelta(list(model.parameters()), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
    if optimizer == "Adagrad":
        optimizer = optim.Adagrad(list(model.parameters()), lr=learning_rate, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    if optimizer == "AdamW":
        optimizer = optim.AdamW(list(model.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    if optimizer == "Adamax":
        optimizer = optim.Adamax(list(model.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if optimizer == "ASGD":
        optimizer = optim.ASGD(list(model.parameters()), lr=learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    if optimizer == "Rprop":
        optimizer = optim.Rprop(list(model.parameters()), lr=learning_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    if optimizer == "RMSprop":     
        optimizer = optim.RMSprop(list(model.parameters()), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        
    import torch.optim as optim
    import numpy as np
    import numpy
   # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    gen = Generator()
    import random
    criterion = nn.L1Loss()
    for epoch in range(1,6):  # loop over the dataset multiple times
        running_loss = 0.0
        i = 0
        
        for i in range(0,1000):
        # get the inputs; data is a list of [inputs, labels]
            k=0
            file_no = random.randint(0, 16000)
            inputs,labels = get_data_batch(file_no)
            INN = inputs.to(device)
            OUT = labels.to(device)
            optimizer.zero_grad()
            outputs = model(INN)
            loss = criterion(outputs, OUT)
            loss.backward()
            optimizer.step()
            del INN
            del OUT
            torch.cuda.empty_cache() 
            # print statistics
            running_loss += loss.item()
            LOSS = 0.0
            if i == 999:    # print every 2000 mini-batches# CHANGE THIS VALUE
               # print('[%d, %5d] loss: %.5f' %(epoch , i + 1, running_loss / 3500))
                running_loss = 0.0
            
                g = 0
                for g in range(0,312):
                    k=0
                    file_no = random.randint(67012, 72730)
		
                    inputs,labels = get_data_batch(file_no)
                    INN = inputs.to(device)
                    OUT = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(INN)
                    loss = criterion(outputs, OUT)
                    LOSS  += loss.item()
                    del INN
                    del OUT
                    torch.cuda.empty_cache() 
                                     
                LOSS = LOSS/312
                running_loss = 0.0
                if epoch ==1 :
                    best_loss = LOSS
                  #  torch.save(net, 'pca_30_points_relational_model_2.pt')
                if LOSS < best_loss:
                    best_loss = LOSS
                  #  torch.save(net, 'pca_30_points_relational_model_2.pt')
                del LOSS
                del running_loss
                torch.cuda.empty_cache() 
                LOSS = 0    
                running_loss = 0.0   
  #  best_loss = best_loss.cpu()    
  #  best_loss = best_loss.detach().numpy()
    return best_loss





def create_model(num_unit1,num_unit2,drop_out):
    model = Net(num_unit1,num_unit2,drop_out)
    model.to(device)
    return model




def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    num_unit1 = trial.suggest_categorical('num_unit1', [256,512,1024, 2048, 4096])
    num_unit2 = trial.suggest_categorical('num_unit2', [256,512,1024, 2048, 4096])   
    learning_rate = trial.suggest_uniform('learning_rate', 0.000000001, 0.001)      
    drop_out      = trial.suggest_uniform('drop_out', 0.00000001, 0.9)    
    optimizer     = trial.suggest_categorical('optimizer', ["SGD","Adam","Adadelta","Adagrad","AdamW","Adamax","ASGD","Rprop","RMSprop"])
    # create the model               
    model = create_model(num_unit1,num_unit2,drop_out)
    # getting the training loss
    error = trainloop(model,learning_rate,optimizer)
    del model
    torch.cuda.empty_cache() 
    return error  # An objective value linked with the Trial object.

study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='minimize')  # Create a new study.
#study = optuna.create_study(direction='minimize')  # Create a new study.
study.optimize(objective, n_trials=50)  # Invoke optimization of the objective function.


df = study.trials_dataframe()
assert isinstance(df, pd.DataFrame)
assert df.shape[0] == 50  # n_trials.

df.to_csv('log.csv')

optuna.visualization.plot_parallel_coordinate(study, params=["num_unit1", "num_unit2","learning_rate","drop_out","optimizer"])


