from __future__ import annotations
from ast import List
from artificial_neural_network import ANN, GM, actFunc, layerInfo, lossFunc, poolMethod, recurrentLinkMode, recurrentTimeWeightModel, typeOfLayer
import auxiliary_tools
from gradMatrix import gradMatrix as gm
import torch
import gradMatrix
from optimization import trainByAdamStep, trainBySameStep

def main():
    if __name__ != "__main__":
        return 
    torch.set_default_device('cuda')
#======================================================================================================================================================
    
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

#======================================================================================================================================================
main()


#梯度测试
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
#感知机 
#============================================================================================================================================
  
    #简版 acc:96.34%
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
'''
    #批归一化版 acc:96.48%
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
'''


#深度残差感知机
#============================================================================================================================================

    #简版 acc:97.03%
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
'''
    
    #批归一化版 acc:96.35%
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
'''



#卷积神经网络
#============================================================================================================================================
   
    #leNet-5 acc:98.86%
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
'''

    #单尺寸卷积核 acc:98.88%
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
'''

    #多尺寸卷积核
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    cTable=[[0,2,4,6,8],[1,3,5,7,9]]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[7,3],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[3,7],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cnn_multiple',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.00001,minRound=1)

'''

#卷积残差网络
#============================================================================================================================================
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]],aFunc=[actFunc.Non,actFunc.Non]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization,aFunc=[actFunc.LReLU]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]],aFunc=[actFunc.Non,actFunc.Non]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization,aFunc=[actFunc.LReLU]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]],aFunc=[actFunc.Non,actFunc.Non]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization,aFunc=[actFunc.LReLU]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]],aFunc=[actFunc.Non,actFunc.Non]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization,aFunc=[actFunc.LReLU]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]],aFunc=[actFunc.Non,actFunc.Non]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization,aFunc=[actFunc.LReLU]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]],aFunc=[actFunc.Non,actFunc.Non]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization,aFunc=[actFunc.LReLU]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[5,5],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_res_softMax',batchSize=20)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.00001)
'''
   #ADD
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4,7,9,11,13}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,5,7,9,11}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={3,5,7,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4,6}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resAdd_softMax',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.00001,minRound=1)
'''


   #CONCATE
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[4],trainDS.x.shape[5]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4,7,9,11,13}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,5,7,9,11}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={3,5,7,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4,6}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[3]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resCon_softMax',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.00001,minRound=1)
'''

   #AddConcate
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4,6,9,11,13,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4,6,9,11,13,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4,7,9,11,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4,7,9,11,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,5,7,9,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,5,7,9,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={3,5,7,10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={3,5,7,10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={3}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={3}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[15],kernelSize=[[3,3]]))  
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[15],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[40]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resAddCon_softMax',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.00001)

'''

#循环神经网络
#============================================================================================================================================
    #感知机版 acc:95.18%
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
    ann=ANN(modelName='rnn_perceptron',batchSize=1,numOfMaxRecurrent=numOfTime,recurrentTimeWeightModel=recurrentTimeWeightModel.classifier)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.001,minRound=1)
'''
    #感知机_批量归一化版 acc:97.19 %
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
    trainByAdamStep(ann,trainDS,testDS,round=20,step=0.001,minRound=1)
'''

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
'''

    #感知机_批量归一化版 acc:96.88 %
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
'''