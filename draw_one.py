import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
graph = 'CNN'
modelType1='FedClar'
modelType2 ='MDS-FL'
filename1='FEDAVG_LR_4LE_100CR_LSTM_Fedclar_0.4_画图'
filename2 = 'FEDAVG_LR_4LE_100CR_LSTM_Fedclar__adam1'
dataset='sisfall'
# dataset='REALWORD'

if(dataset=='sisfall'):
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


# trainLossHistory1 = hkl.load(filepath1+'trainLossHistory.hkl')
# valiLossHistory1 = hkl.load(filepath1+'valiLossHistory.hkl')
# stdValiLossHistory1 = hkl.load(filepath1+'stdValiLossHistory.hkl')
# stdTrainLossHistory1 = hkl.load(filepath1+'stdTrainLossHistory.hkl')
trainLossHistory2 = hkl.load(filepath2+'trainLossHistory.hkl')
valiLossHistory2 = hkl.load(filepath2+'valiLossHistory.hkl')
stdValiLossHistory2 = hkl.load(filepath2+'stdValiLossHistory.hkl')
stdTrainLossHistory2 = hkl.load(filepath2+'stdTrainLossHistory.hkl')

print(trainLossHistory1)
epoch_range = range(1, 100+1)
# plt.errorbar(epoch_range, trainLossHistory1,  label='FedClarTrainLoss',c="red",marker = '.',markevery=5,linewidth=2.0)
# plt.plot(epoch_range, trainLossHistory1, markevery=[np.argmax(trainLossHistory1)], ls="" )
# plt.errorbar(epoch_range, valiLossHistory1,  label='FedClarValiLoss', linewidth=2.0, c="red")
# plt.plot(epoch_range, valiLossHistory1, markevery=[np.argmax(valiLossHistory1)], ls="")

plt.errorbar(epoch_range, trainLossHistory2,  label='MDS-FLTrainLoss',c="blue",markevery=5,marker = '.',linewidth=1.0)
plt.plot(epoch_range, trainLossHistory2, markevery=[np.argmax(trainLossHistory2)], ls="")
plt.errorbar(epoch_range, valiLossHistory2,  label='MDS-FLValiLoss',  c="blue",linewidth=1.0)
plt.plot(epoch_range, valiLossHistory2, markevery=[np.argmax(valiLossHistory2)], ls="")

saveGraph(dataset+'_'+graph+"Loss", "Loss", legendLoc='upper right')
