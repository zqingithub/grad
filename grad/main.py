from __future__ import annotations
from ast import List
from artificial_neural_network import ANN, GM, actFunc, layerInfo, lossFunc, poolMethod, recurrentLinkMode, recurrentTimeWeightModel, typeOfLayer
import auxiliary_tools
from gradMatrix import gradMatrix as gm
import torch
import gradMatrix
from optimization import trainByAdamStep, trainBySameStep
import random

def main():
    if __name__ != "__main__":
        return 
    torch.set_default_device('cuda')
#======================================================================================================================================================
    #粘贴的代码须作为main函数的部分代码，因此粘贴的代码开头需要与'#'对齐
    #粘贴的代码开头须与开头的'#'号对齐

    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70],aFunc=[actFunc.GELU]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron',batchSize=50)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
#======================================================================================================================================================
main()


#============================================================================================================================================
#模型导出与导入
#============================================================================================================================================
    
    #模型导出
'''
    auxiliary_tools.modelStructureSaveToJson(ann,"C:\\")
    auxiliary_tools.modelParameterSaveToFile(ann,"C:\\")
'''

    #模型导入
'''
    model=auxiliary_tools.createModelStructureFromJson(file='C:\\perceptron-structure-2026-02-02_15-02-55.json')
    auxiliary_tools.loadModelParameterFromFile(model,'C:\\perceptron-parameter-2026-02-02_15-02-55.pth')
'''

    #模型测试
'''
    trainDS,testDS=auxiliary_tools.getDataSet()

    model.createFinishModel()
    model.updateParaToFinishModel()
    print("testDate_acc:",auxiliary_tools.computeAcc(model,testDS,testDS.x.shape[0],isUseFinishModel=False)*100,"%")
    if model.batchSize>1:
        isHaveBatchNormalization=False
        for i in model.layerInfo:
            if i.typeOfLayer==typeOfLayer.batchNormalization:
                isHaveBatchNormalization=True
        if isHaveBatchNormalization:
            print("testDate_acc(not use train data to compute bnPara):",auxiliary_tools.computeAcc(model,testDS,testDS.x.shape[0],isUseFinishModel=True)*100,"%")
            if model.numOfMaxRecurrent>1:
                print("testDate_acc(not use train data to compute bnPara,not use per time bnPara):",auxiliary_tools.computeAcc(model,testDS,testDS.x.shape[0],isUseFinishModel=True,isUseBnParaPerTime=False)*100,"%")
            model.updateParaToFinishModel(isUseTrainData=True,trainData=trainDS.x)
            print("testDate_acc(use train data to compute bnPara):",auxiliary_tools.computeAcc(model,testDS,testDS.x.shape[0],isUseFinishModel=True)*100,"%")
            if model.numOfMaxRecurrent>1:
                print("testDate_acc(use train data to compute bnPara,not use per time bnPara):",auxiliary_tools.computeAcc(model,testDS,testDS.x.shape[0],isUseFinishModel=True,isUseBnParaPerTime=False)*100,"%")
'''
#============================================================================================================================================


#============================================================================================================================================
#梯度测试
#============================================================================================================================================
'''
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[4],shortCircuitAddSet={4},shortCircuitConcateSet={4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[3],isRecurrent=True,rcLinkMode=recurrentLinkMode.addCon))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5],isRecurrent=True,rcLinkMode=recurrentLinkMode.add))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[2]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='test',batchSize=10,numOfMaxRecurrent=6,isBidirectional=True)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2,isCheckShare=False)
    x=torch.rand([10,4,1,1,2,2],dtype=torch.float)
    y=torch.rand([10,4,1,2],dtype=torch.float)
    ann.setXBatch(x,0)
    ann.setRealYBatch(y,0)
    ann.computeValue()
    ann.computeGrad()
    temp=[]
    for i in ann.modelPara:
        print(i.grad)
        temp.append(i.grad.clone())
    print('==============================')
    ann.computeGradByEstimate(step=0.0001)
    c=0
    for i in ann.modelPara:
        print(i.grad)
        temp[c]=temp[c]-i.grad
        c+=1
    print('======================================================================================')
    for i in temp:
        print(i)
'''
#============================================================================================================================================


#============================================================================================================================================
#感知机 
#============================================================================================================================================
  
    #简版 prediction accuracy‌:97.41%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron',batchSize=50)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #批归一化版 prediction accuracy‌:97.48%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[90]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[80]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron_batchNorm',batchSize=50)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
#============================================================================================================================================


#============================================================================================================================================
#深度残差感知机
#============================================================================================================================================

    #简版 prediction accuracy‌:97.66%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4,6,8,10,12,14,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4,6,8,10,12,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4,6,8,10,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4,6,8,10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4,6,8}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4,6}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron_deep_res',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
    
    #批归一化版 prediction accuracy‌:97.89%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,5,7,9,11,14,16,18}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={3,5,7,9,12,14,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={3}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4,6,9,11,13}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4,7,9,11}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,5,7,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={3}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={3,5,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2,4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200],shortCircuitAddSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[200]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron_deep_res_BN',batchSize=50)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
#============================================================================================================================================


#============================================================================================================================================
#卷积神经网络
#============================================================================================================================================
   
    #leNet-5 prediction accuracy‌:99.00%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[16],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='leNet-5',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainBySameStep(ann,trainDS,testDS,round=20,step=0.1,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #单尺寸卷积核 prediction accuracy‌:99.05%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[12],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[12],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[150]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cnn_single',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainBySameStep(ann,trainDS,testDS,round=20,step=0.01,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #多尺寸卷积核 prediction accuracy‌:98.82%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6,6,6],kernelSize=[[3,3],[5,5],[7,7]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6,6,6],kernelSize=[[3,3],[5,5],[7,7]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    cTable=[[0,2,4,6,8,10,12,14,16],[1,3,5,7,9,11,13,15,17]]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[8,8],kernelSize=[[3,3],[5,5]],channelTable=cTable))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[8,8],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cnn_multiple',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
#============================================================================================================================================


#============================================================================================================================================
#卷积残差网络
#============================================================================================================================================
   
    #ADD prediction accuracy‌:99.22%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={2,4,7,9,12,14,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={2,5,7,10,12,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={3,5,8,10,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitAddSet={2,5,7,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitAddSet={3,5,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitAddSet={2,4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitAddSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resAdd',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #CONCATE prediction accuracy‌:98.97%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitConcateSet={2,4,7,9,12,14,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitConcateSet={2,5,7,10,12,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitConcateSet={3,5,8,10,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitConcateSet={2,5,7,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitConcateSet={3,5,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitConcateSet={2,4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resCon',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #AddConcate prediction accuracy‌:99.35%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={2,4,7,9,12,14,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitConcateSet={2,5,7,9,12,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={2,5,7,10,12,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitConcateSet={3,5,7,10,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={3,5,8,10,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitConcateSet={2,4,7,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitAddSet={2,5,7,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitConcateSet={2,5,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitAddSet={3,5,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitConcateSet={3,5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitAddSet={2,4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitAddSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resAddCon',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #AddConcate批归一化版 prediction accuracy‌:98.35%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={3,5,8,11,14,17,19}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitConcateSet={3,6,9,11,14,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={2,5,8,11,14,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitConcateSet={3,6,8,11,13}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[7,7]],shortCircuitAddSet={3,6,9,12,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitConcateSet={3,5,8,10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitAddSet={3,6,9,11}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitConcateSet={2,5,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitAddSet={3,6,8}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]],shortCircuitConcateSet={3,5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitAddSet={3,5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitConcateSet={3}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]],shortCircuitAddSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resAddCon__batchNorm',batchSize=50)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=15,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
#============================================================================================================================================


#============================================================================================================================================
#单向循环神经网络
#============================================================================================================================================

    #感知机版 prediction accuracy‌:95.42%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=4
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[40],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn_perceptron',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=5,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #感知机_批量归一化版 prediction accuracy‌:96.35%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=4
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[90]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[80],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn_perceptron_bn',batchSize=50,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=5,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
    
    #卷积神经网络版 acc:97.84 %
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=4
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6],kernelSize=[[5,3]],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[16],kernelSize=[[5,3]],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn_cov',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainByAdamStep(ann,trainDS,testDS,round=5,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #卷积神经网络版(时序特征有重叠，使用特征滑动窗口) acc:98.44 %
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=5
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime,width=20,step=2)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime,width=20,step=2)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6],kernelSize=[[7,5]],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[16],kernelSize=[[5,3]],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn_cov_slide',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainByAdamStep(ann,trainDS,testDS,round=5,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
#============================================================================================================================================


#============================================================================================================================================
#双向循环神经网络
#============================================================================================================================================
    
    #感知机版 acc:96.18%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=4
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn_Bi_perceptron',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier,isBidirectional=True)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #感知机_批量归一化版 acc:97.53 %
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=4
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[90]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[80],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn_Bi_perceptron_bn',batchSize=50,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier,isBidirectional=True)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #卷积神经网络版 acc:97.96 %
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=4
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6],kernelSize=[[5,3]],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[16],kernelSize=[[5,3]],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn_Bi_cov',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier,isBidirectional=True)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #卷积神经网络版(时序特征有重叠，使用特征滑动窗口) acc:98.08 %
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=5
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime,width=20,step=2)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime,width=20,step=2)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6],kernelSize=[[7,5]],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[16],kernelSize=[[5,3]],isRecurrent=True))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.norm]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn_Bi_cov_slide',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier,isBidirectional=True)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.L2)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
#============================================================================================================================================


#============================================================================================================================================
#单向LSTM循环神经网络
#============================================================================================================================================
    
    #感知机版 acc:94.15%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=4
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.LSTM))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn-LSTM_perceptron',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=2.5,step=0.001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #卷积神经网络版(时序特征有重叠，使用特征滑动窗口) acc:97.81%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=13
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime,width=16,step=1)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime,width=16,step=1)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[12],kernelSize=[[7,7]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.LSTM))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.LSTM))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[8],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn-LSTM_cov_slide',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
#============================================================================================================================================


#============================================================================================================================================
#双向LSTM循环神经网络
#============================================================================================================================================
    
    #感知机版 acc:96.58%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=4
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[70]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.LSTM))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn-LSTM_Bi_perceptron',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier,isBidirectional=True)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=2,step=0.0005,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''

    #卷积神经网络版(时序特征有重叠，使用特征滑动窗口) acc:97.45%
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    numOfTime=13
    auxiliary_tools.changeToTimeSequence(trainDS,numOfTime,width=16,step=1)
    auxiliary_tools.changeToTimeSequence(testDS,numOfTime,width=16,step=1)
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[12],kernelSize=[[7,7]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.LSTM))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.LSTM))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[8],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='rnn-LSTM_Bi_cov_slide',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier,isBidirectional=True)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.0001,minRound=1)
    auxiliary_tools.modelStructureSaveToJson(ann)
    auxiliary_tools.modelParameterSaveToFile(ann)
'''
#============================================================================================================================================