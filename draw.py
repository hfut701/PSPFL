import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
graph = 'RNN'
modelType1='FedClar'
modelType2 ='Fed-Sim'
# realworld
# FEDAVG_LR_3LE_100CR_CNN_Fedclar_0.4
# FEDAVG_LR_3LE_100CR_CNN_Fed_wei_personalized_former_nodelete_f_test_0_1_0.5_lossnew
# FEDAVG_LR_4LE_100CR_LSTM_Fedclar_0.4_画图
# FEDAVG_LR_4LE_100CR_LSTM_Fed_wei_personalized_former_nodelete_f_test_0_1_0.5_lossnew

# sisfall
#FEDAVG_LR_3LE_100CR_CNN_Fedclar_0.4
#FEDAVG_LR_3LE_100CR_CNN_Fed_wei_personalized_former_nodelete_f_test_0_1_0.5_loss
filename1='FEDAVG_LR_4LE_100CR_LSTM_Fedclar_0.4_画图'
filename2 = 'FEDAVG_LR_4LE_100CR_LSTM_Fed_wei_personalized_former_nodelete_f_test_0_1_0.5_losstest2'
dataset='SisFall'

if(dataset=='SisFall'):
    filepath1 = 'savedModels/'+filename1+'/sisfall/trainingStats/'
    filepath2 = 'savedModels/'+filename2+'/sisfall/trainingStats/'

else:
    filepath1 = 'savedModels/' + filename1 + '/REALWORLD_CLIENT/trainingStats/'
    filepath2 = 'savedModels/' + filename2 + '/REALWORLD_CLIENT/trainingStats/'



def saveGraph(title="", accuracyOrLoss="Loss", legendLoc='lower right'):
    plt.title(title)
    plt.ylabel(accuracyOrLoss)
    plt.xlabel('Communication Round')
    plt.legend(loc=legendLoc)
    plt.savefig(title.replace(" ", "") + '.png', dpi=100)
    plt.clf()
trainLossHistory1 = []
valiLossHistory1 = []
stdTrainLossHistory1= []
stdValiLossHistory1 = []
trainLossHistory2 = []
valiLossHistory2 = []
stdTrainLossHistory2 = []
stdValiLossHistory2 = []


trainLossHistory1 = hkl.load(filepath1+'trainLossHistory.hkl').tolist()
valiLossHistory1 = hkl.load(filepath1+'valiLossHistory.hkl').tolist()
# stdValiLossHistory1 = hkl.load(filepath1+'stdValiLossHistory.hkl')
# stdTrainLossHistory1 = hkl.load(filepath1+'stdTrainLossHistory.hkl')
trainLossHistory2 = hkl.load(filepath2+'trainLossHistory.hkl')
valiLossHistory2 = hkl.load(filepath2+'valiLossHistory.hkl')
# stdValiLossHistory2 = hkl.load(filepath2+'stdValiLossHistory.hkl')
# stdTrainLossHistory2 = hkl.load(filepath2+'stdTrainLossHistory.hkl')
trainLossHistory3=[]
valiLossHistory3 =[]
for i in range(len(trainLossHistory1)):
    if i%2==0:
        trainLossHistory3.append(trainLossHistory1[i])
        valiLossHistory3.append(valiLossHistory1[i])
print(trainLossHistory2)
print(len(trainLossHistory2))
epoch_range = range(1, 100+1)
plt.errorbar(epoch_range, trainLossHistory1,  label='FedClarTrainLoss',c="red",marker = '.',markevery=5,linewidth=1.0)
plt.plot(epoch_range, trainLossHistory1, markevery=[np.argmax(trainLossHistory1)], ls="" )
plt.errorbar(epoch_range, valiLossHistory1,  label='FedClarValiLoss', linewidth=2.0, c="red")
plt.plot(epoch_range, valiLossHistory1, markevery=[np.argmax(valiLossHistory3)], ls="")

plt.errorbar(epoch_range, trainLossHistory2,  label='Fed-SimTrainLoss',c="blue",markevery=5,marker = '.',linewidth=1.0)
plt.plot(epoch_range, trainLossHistory2, markevery=[np.argmax(trainLossHistory2)], ls="")
plt.errorbar(epoch_range, valiLossHistory2,  label='Fed-SimValiLoss',  c="blue",linewidth=2.0)
plt.plot(epoch_range, valiLossHistory2, markevery=[np.argmax(valiLossHistory2)], ls="")

saveGraph(dataset+'_'+graph+"Loss", "Loss", legendLoc='upper right')
