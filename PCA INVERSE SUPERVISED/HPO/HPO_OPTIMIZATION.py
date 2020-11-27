import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import pandas as pd
import numpy  as np
import pandas
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):

    def __init__(self,num_unit1):
        super(Network, self).__init__()
        self.Dense1   =  nn.Linear(30,num_unit1)           
        self.out      =  nn.Linear(num_unit1,42)

        
    
    def forward(self, x):
        #branch one
        x   = self.Dense1(x)
        x   = F.leaky_relu(x, 0.1, inplace=True)
        x   = self.out(x)
        x   = F.relu(x)
        ret = x.view(-1, 42)
        return ret



y = pd.read_csv('INVERSE_14_PCA_FILE.csv',index_col=None)
x = pd.read_csv('30_POINT_RELATION_FILE.csv',index_col=None)



x_train = x.loc[0:6000]
y_train = y.loc[0:6000]

x_VAL = x.loc[60000:72757]
y_VAL = y.loc[60000:72757]


X_VALIDATION = np.array(x_VAL).tolist()
X_VALIDATION = torch.FloatTensor(X_VALIDATION)
X_VALIDATION = X_VALIDATION.to(device)

Y_VALIDATION = np.array(y_VAL).tolist()
Y_VALIDATION = torch.FloatTensor(Y_VALIDATION)
Y_VALIDATION = Y_VALIDATION.to(device)

def trainloop(Inverse_pca_pytorch,learning_rate):

    import random
    optimizer = torch.optim.SGD(Inverse_pca_pytorch.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.L1Loss()
    for epoch in range(1,16):
         running_loss = 0.0
    # loop over the dataset multiple times
         for i in range (1,1876):
            number  = random.randint(0,59984)
            input1_1  = x.iloc[number:number+1]
            output1_1 = y.iloc[number:number+1]
        
            b1 = np.array(output1_1)
            b  = np.array(input1_1)
    # getting whole batch in the loop below        
            for u in range (1,32):
                a = x.iloc[number+u:number+u+1]
                a1= y.iloc[number+u:number+u+1]
                a = np.array(a)
                a = np.reshape(a, (1,30))
                a1 = np.array(a1)
                a1 = np.reshape(a1, (1,42))
            
                b  = np.concatenate((a, b), axis=0)
                b1 = np.concatenate((a1, b1), axis=0)
                

            input1 = np.array(b)
            input1 = np.array(input1).tolist()
            input1 = torch.FloatTensor(input1)
            input1 = input1.to(device)

            output1 = np.array(b1)
            output1 = np.array(output1).tolist()
            output1 = torch.FloatTensor(output1)        
            output1 = output1.to(device)
        
            optimizer.zero_grad()
            outputs_train = Inverse_pca_pytorch(input1)        
            
                     
            
            loss = criterion(outputs_train,output1)
            loss.backward()

            del input1
            del output1
            del outputs_train
            optimizer.step()
            running_loss += loss
            if i == 1875:
                PREDICTION = Inverse_pca_pytorch(X_VALIDATION)
                LOSS       = criterion(PREDICTION, Y_VALIDATION)
        
                if epoch ==1 :
                    best_loss = LOSS
                    torch.save(Inverse_pca_pytorch, 'best_architecture.pt')
                if LOSS < best_loss:
                    best_loss = LOSS
                    torch.save(Inverse_pca_pytorch, 'best_architecture.pt')
                LOSS = LOSS.cpu()
                running_loss = running_loss.cpu()
            #    print('[%d, validation loss %.5f] training loss %.7f '%(epoch , LOSS , (running_loss/3749).detach().numpy() ))
                del PREDICTION
                del LOSS
                LOSS = 0 
                torch.cuda.empty_cache()
    best_loss = best_loss.cpu()      
    best_loss = best_loss.detach().numpy()
    return best_loss


def create_model(num_unit1):
    model = Network(num_unit1)
    model.to(device)
    return model



def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    num_unit1 = trial.suggest_int('num_unit1', 256,8096,512)
    learning_rate = trial.suggest_uniform('learning_rate', 0.00000001, 0.001)      
    # create the model               
    model = create_model(num_unit1)
    # getting the training loss
    error = trainloop(model,learning_rate)
    torch.cuda.empty_cache()
    del model
    return error  # An objective value linked with the Trial object.

study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='minimize')  # Create a new study.
#study = optuna.create_study(direction='minimize')  # Create a new study.
study.optimize(objective, n_trials=30)  # Invoke optimization of the objective function.

df = study.trials_dataframe()
assert isinstance(df, pd.DataFrame)
assert df.shape[0] == 30  # n_trials.

df.to_csv('1_layer_log.csv')

optuna.visualization.plot_parallel_coordinate(study, params=["num_unit1","learning_rate"])

