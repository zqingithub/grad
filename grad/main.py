from __future__ import annotations
from ast import List
from artificial_neural_network import ANN, GM, actFunc, layerInfo, lossFunc, poolMethod, typeOfLayer
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
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[3],trainDS.x.shape[4]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[60]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[2]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron',batchSize=2)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=5,step=0.001)
    
    '''
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[1],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[2]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron_softmax_cov',batchSize=2)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    ann.x[0].value=torch.rand([1,1,5,5],dtype=torch.float)
    ann.realY[0].value=torch.tensor([[0.2,0.8]])
    ann.x[1].value=torch.rand([1,1,5,5],dtype=torch.float)
    ann.realY[1].value=torch.tensor([[0.7,0.3]])
    ann.computeValue()
    ann.computeGrad()
    temp=[]
    for i in ann.modelPara:
        print(i.grad)
        temp.append(i.grad.clone())
    print('=====')
    ann.lossValue.computeGradByEstimate(ann.modelPara,0.01)
    c=0
    for i in ann.modelPara:
        print(i.grad)
        temp[c]=temp[c]-i.grad
        c+=1
    print('===============================')
    for i in temp:
        print(i)
    '''
    

#======================================================================================================================================================
main()


#梯度测试
'''
    infoOfLayer:List[layerInfo]=[None]*3
    infoOfLayer[0]=layerInfo(layerType=typeOfLayer.input,outputDim=[1,2])
    infoOfLayer[1]=layerInfo(layerType=typeOfLayer.link,outputDim=[1,4])
    infoOfLayer[2]=layerInfo(layerType=typeOfLayer.link,outputDim=[1,3],aFunc=actFunc.softMax)
    ann=ANN(modelName='perceptron')
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)

    ann.setX(torch.tensor([[2.0,5.0]]))
    ann.setRealY(torch.tensor([[0.2,0.3,0.5]]))
    ann.computeValue()
    ann.computeGrad()
    for i in ann.modelPara:
        print(i.grad)
    print('=========================')
    ann.computeValue()
    ann.lossValue.computeGradByEstimate(ann.modelPara,0.01)
    for i in ann.modelPara:
        print(i.grad)
'''
#感知机
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[60]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron',batchSize=100)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=5,step=0.001)
'''
    #批归一化版
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[60]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[60]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[60]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron_batchNorm',batchSize=50)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=5,step=0.001)
'''


#深度残差感知机
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitAddSet={4,9,14},shortCircuitConcateSet={11}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={4,9,14},shortCircuitAddSet={9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitAddSet={4,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={4,9}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitAddSet={4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={4}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron_deep_res',batchSize=2)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.0001)
'''
    #批归一化版

'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitAddSet={5,10,15},shortCircuitConcateSet={12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={5,10,15},shortCircuitAddSet={10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitAddSet={4,10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={4,10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitAddSet={5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100],shortCircuitConcateSet={5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.batchNormalization))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[100]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='perceptron_deep_res_BN',batchSize=50)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.0001)
'''



#leNet-5
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[6],kernelSize=[[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    cTable=[[0,1,2],[1,2,3],[2,3,4],[3,4,5],[0,4,5],[0,1,5],[0,1,2,3],[1,2,3,4],[2,3,4,5],[0,3,4,5],[0,1,4,5],[0,1,2,5],[0,1,3,4],[1,2,4,5],[0,2,3,5],[0,1,2,3,4,5]]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[16],kernelSize=[[5,5]],channelTable=cTable))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[120]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='leNet-5',batchSize=5)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.0001)
'''

  #2

'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    cTable=[[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9],[7,8,9,0],[8,9,0,1],[9,0,1,2],[0,2,4,6],[2,4,6,8],[1,3,5,7],[3,5,7,9],[0,2,4,6,8],[1,3,5,7,9],\
            [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9],[7,8,9,0],[8,9,0,1],[9,0,1,2],[0,2,4,6],[2,4,6,8],[1,3,5,7],[3,5,7,9],[0,2,4,6,8],[1,3,5,7,9]]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[16,16],kernelSize=[[3,3],[5,5]],channelTable=cTable))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[15],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[15],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[40]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov2#_softMax',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.0001)

'''

#卷积残差网络
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
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4,6,9,11,13,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4,7,9,11,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,5,7,9,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={3,5,7,10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,4,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={2,5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitAddSet={3}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[15],kernelSize=[[3,3]]))  
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[15],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[40]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resAdd_softMax',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.00001)
'''


   #CONCATE
'''
    trainDS,testDS=auxiliary_tools.getDataSet()
    infoOfLayer:List[layerInfo]=[]
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.input,channelSize=[1],kernelSize=[[trainDS.x.shape[2],trainDS.x.shape[3]]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4,6,9,11,13,16}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4,7,9,11,14}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,5,7,9,12}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={3,5,7,10}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,4,7}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={2,5}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]],shortCircuitConcateSet={3}))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[10,10],kernelSize=[[3,3],[5,5]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.maxPool],kernelSize=[[2,2]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[15],kernelSize=[[3,3]]))  
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[15],kernelSize=[[3,3]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[40]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.conv,channelSize=[trainDS.y.shape[1]],aFunc=[actFunc.softMax]))
    infoOfLayer.append(layerInfo(layerType=typeOfLayer.output))
    ann=ANN(modelName='cov_resCon_softMax',batchSize=1)
    ann.createAnn(lyInfo=infoOfLayer,lsFunc=lossFunc.crossEntropy,isCheckShare=False)
    trainByAdamStep(ann,trainDS,testDS,round=10,step=0.00001)
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

'''
torch.set_default_device('cuda')
layer=[aNNetwork.layer]*5
layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=16)
layer[1]=aNNetwork.layer(layer=tpLayer.conv,kernelSize=2,channelSize=2)
layer[2]=aNNetwork.layer(layer=tpLayer.conv,kernelSize=2,channelSize=3)
layer[2].channelTable=[[1,1],[0,1],[1,0]]
layer[3]=aNNetwork.layer(layer=tpLayer.pool,kernelSize=2)
layer[4]=aNNetwork.layer(layer=tpLayer.link,outputDim=6)
cnn=aNNetwork.CNN(layer)
cnn.x.value=torch.rand(1,16,dtype=torch.float)
cnn.realY.value=torch.rand(1,6,dtype=torch.float)
cnn.lossValue.computeValue()
cnn.lossValue.computeGrad(cnn.paraModel)
for i in cnn.paraModel:
    print(i.grad)
print("======================================")
cnn.lossValue.computeGradByEstimate(cnn.paraModel,0.01)
for i in cnn.paraModel:
    print(i.grad)
'''

'''
torch.set_default_device('cuda')
temp1=gm(torch.tensor([[1,1,1],[1,1,1],[1,1,1]]),gradMatrix.variableType.gradVariable)
temp2=gm(torch.tensor([[2,2,2],[2,2,2],[2,2,2]]))
twinDict={temp2:temp2}
temp3=temp1+temp2
temp5=temp3+temp3
temp3=temp5
temp4=gm.copy(temp3,twinsDict=twinDict)
temp2.value[1,1]=4
twinDict[temp2].value[1,1]=5
temp3.computeValue()
temp4.computeValue()
print(temp3.value)
print(temp4.value)
'''


'''

torch.set_default_device('cuda')
#temp1=gm(torch.rand((5,5),dtype=torch.float),gradMatrix.variableType.gradVariable)
#temp2=gm(torch.rand((2,2),dtype=torch.float),gradMatrix.variableType.gradVariable)
#temp3=gm(torch.rand((2,3),dtype=torch.float))

temp1=gm(torch.ones(3,3,dtype=torch.float),gradMatrix.variableType.gradVariable)
#temp2=gm(torch.tensor((6,2),dtype=torch.float),gradMatrix.variableType.gradVariable)
temp2=gm(torch.ones(3,3,dtype=torch.float),gradMatrix.variableType.gradVariable)
temp3=gm(torch.ones(3,3,dtype=torch.float))

gV=[temp1,temp2]



#temp1=temp1.conv2D(temp2)
#temp1=temp1.avgPool2D(2)
#temp1=temp1.reshape([1,torch.prod(torch.tensor(temp1.value.shape))])


temp4=temp1+temp2

print("temp4")
temp4.computeValue()
print(temp4.value)

print("=============")
temp4.computeGrad(gV)
gVt1=gV[0].grad.clone()
gVt2=gV[1].grad.clone()
print(gV[0].grad)
print(gV[1].grad)
print("=============")
temp4.computeGradByEstimate(gV,0.01)
print(gV[0].grad)
print(gV[1].grad)
print("=============")
print(gVt1-gV[0].grad)
print(gVt2-gV[1].grad)

'''



'''
torch.set_default_device('cuda')

trainDS,testDS=aNNetwork.getDataSet()
aF=[actFunc.LReLU,actFunc.LReLU,actFunc.LReLU,actFunc.LReLU,actFunc.norm]
aNN=aNNetwork.FNN_res([trainDS.x.shape[1],120,120,120,120,trainDS.y.shape[1]],aF)
aNNetwork.trainBySameStep(aNN,trainDS,testDS,50,0.005,10000000)
'''
'''
#leNet-5
torch.set_default_device('cuda')
trainDS,testDS=aNNetwork.getDataSet()
layer=[aNNetwork.layer]*7
layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=trainDS.x.shape[1])
layer[1]=aNNetwork.layer(layer=tpLayer.conv,kernelSize=5,channelSize=6)
layer[2]=aNNetwork.layer(layer=tpLayer.pool,kernelSize=2,actFunc=actFunc.Non)
channelTable=torch.tensor([[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],\
                            [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],\
                            [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],\
                            [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],\
                            [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],\
                            [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]],dtype=torch.long)
layer[3]=aNNetwork.layer(layer=tpLayer.conv,kernelSize=5,channelSize=16,channelTable=channelTable.T)
layer[4]=aNNetwork.layer(layer=tpLayer.pool,kernelSize=2,actFunc=actFunc.Non)
layer[5]=aNNetwork.layer(layer=tpLayer.link,outputDim=120)
layer[6]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,actFunc=actFunc.norm)
cnn=aNNetwork.CNN(layer)
aNNetwork.trainBySameStep(cnn,trainDS,testDS,50,0.1,2000)
'''
'''
torch.set_default_device('cuda')
trainDS,testDS=aNNetwork.getDataSet()
layer=[aNNetwork.layer]*8
layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=784)
layer[1]=aNNetwork.layer(layer=tpLayer.link,outputDim=40)
layer[2]=aNNetwork.layer(layer=tpLayer.link,outputDim=40,isRecurrent=True,isShortCircuit=True)
layer[3]=aNNetwork.layer(layer=tpLayer.link,outputDim=40)
layer[4]=aNNetwork.layer(layer=tpLayer.link,outputDim=40,isShortCircuit=True)
layer[5]=aNNetwork.layer(layer=tpLayer.link,outputDim=40)
layer[6]=aNNetwork.layer(layer=tpLayer.link,outputDim=40,isRecurrent=True)
layer[7]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,actFunc=actFunc.norm)
rnn=aNNetwork.RNN(layer,numOfMaxRecurrent=1)
aNNetwork.trainBySameStep(rnn,trainDS,testDS,50,0.1,10000000)
'''
'''
torch.set_default_device('cuda')
trainDS,testDS=aNNetwork.getDataSet()
layer=[aNNetwork.layer]*4
layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=196)
layer[1]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,isRecurrent=True)
layer[2]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,isRecurrent=True)
layer[3]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,actFunc=actFunc.norm)
rnn=aNNetwork.RNN(layer,numOfMaxRecurrent=4)
aNNetwork.trainBySameStep(rnn,trainDS,testDS,50,0.1,2000)

'''
'''
torch.set_default_device('cuda')
trainDS,testDS=aNNetwork.getDataSet()
layer=[aNNetwork.layer]*5
layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=3)
layer[1]=aNNetwork.layer(layer=tpLayer.link,outputDim=4,isRecurrent=True,isShortCircuit=True)
layer[2]=aNNetwork.layer(layer=tpLayer.link,outputDim=5,isRecurrent=True)
layer[3]=aNNetwork.layer(layer=tpLayer.link,outputDim=3,isRecurrent=True)
layer[4]=aNNetwork.layer(layer=tpLayer.link,outputDim=2,actFunc=actFunc.norm)
rnn=aNNetwork.RNN_Bi(layer,numOfMaxRecurrent=6)
rnn.xList[0].value=torch.tensor([[0.9,0.71,0.312]],dtype=torch.float)
rnn.xList[1].value=torch.tensor([[0.2,0.422,0.253]],dtype=torch.float)
rnn.xList[2].value=torch.tensor([[0.6,0.367,0.155]],dtype=torch.float)
rnn.xList[3].value=torch.tensor([[0.1,0.788,0.1233]],dtype=torch.float)
rnn.realYList[0].value=torch.tensor([[0.5,0.5]],dtype=torch.float)
rnn.realYList[1].value=torch.tensor([[0.7,0.3]],dtype=torch.float)
rnn.realYList[2].value=torch.tensor([[0.6,0.4]],dtype=torch.float)
rnn.realYList[3].value=torch.tensor([[0.1,0.9]],dtype=torch.float)
rnn.nowRecurrent=2
rnn.computeValue()
print(rnn.getLossValue())
rnn.computeGrad()
for i in rnn.paraModel:
    print(i.grad)

print("========")

rnn.computeGradByEstimate(rnn.paraModel,0.01)
for i in rnn.paraModel:
    print(i.grad)

print("================================================")
rnn.computeValue()
print(rnn.getLossValue())
rnn.computeGrad()
for i in rnn.paraModel:
    print(i.grad)

print("========")

rnn.computeGradByEstimate(rnn.paraModel,0.01)
for i in rnn.paraModel:
    print(i.grad)



'''

'''
torch.set_default_device('cuda')
trainDS,testDS=aNNetwork.getDataSet()
layer=[aNNetwork.layer]*5
layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=196)
layer[1]=aNNetwork.layer(layer=tpLayer.link,outputDim=40,isRecurrent=True,isShortCircuit=True)
layer[2]=aNNetwork.layer(layer=tpLayer.link,outputDim=40)
layer[3]=aNNetwork.layer(layer=tpLayer.link,outputDim=40,isRecurrent=True)
layer[4]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,actFunc=actFunc.norm)
rnn=aNNetwork.RNN_Bi(layer,numOfMaxRecurrent=4)
aNNetwork.trainBySameStep(rnn,trainDS,testDS,50,0.1,20000)
'''
'''
def main():
    if __name__ != "__main__":
        return 
    #leNet-5
    torch.set_default_device('cuda')
    trainDS,testDS=aNNetwork.getDataSet()
    layer=[aNNetwork.layer]*7
    layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=trainDS.x.shape[1])
    layer[1]=aNNetwork.layer(layer=tpLayer.conv,kernelSize=5,channelSize=6)
    layer[2]=aNNetwork.layer(layer=tpLayer.pool,kernelSize=2,actFunc=actFunc.Non)
    channelTable=torch.tensor([[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],\
                                [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],\
                                [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],\
                                [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],\
                                [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],\
                                [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]],dtype=torch.long)
    layer[3]=aNNetwork.layer(layer=tpLayer.conv,kernelSize=5,channelSize=16,channelTable=channelTable.T)
    layer[4]=aNNetwork.layer(layer=tpLayer.pool,kernelSize=2,actFunc=actFunc.Non)
    layer[5]=aNNetwork.layer(layer=tpLayer.link,outputDim=120)
    layer[6]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,actFunc=actFunc.norm)
    cnn=aNNetwork.CNN(layer,modelName='leNet-5(CNN)1')
    aNNetwork.trainByAdamStep(cnn,trainDS,testDS,10,50000)

main()
'''
'''

def main():
    if __name__ != "__main__":
        return 
    torch.set_default_device('cuda')
    trainDS,testDS=aNNetwork.getDataSet()
    layer=[aNNetwork.layer]*3
    layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=196)
    layer[1]=aNNetwork.layer(layer=tpLayer.link,outputDim=60,isRecurrent=True)
    layer[2]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,actFunc=actFunc.norm)
    rnn=aNNetwork.RNN_Bi(layer,numOfMaxRecurrent=4)
    aNNetwork.trainByAdamStep(rnn,trainDS,testDS,10,50000)

main()

'''

'''
def main():
    if __name__ != "__main__":
        return 
    torch.set_default_device('cuda')
    trainDS,testDS=aNNetwork.getDataSet()
    layer=[aNNetwork.layer]*8
    layer[0]=aNNetwork.layer(layer=tpLayer.input,outputDim=196)
    layer[1]=aNNetwork.layer(layer=tpLayer.link,outputDim=20,isShortCircuit=True)
    layer[2]=aNNetwork.layer(layer=tpLayer.link,outputDim=20,isRecurrent=True)
    layer[3]=aNNetwork.layer(layer=tpLayer.link,outputDim=20,isShortCircuit=True)
    layer[4]=aNNetwork.layer(layer=tpLayer.link,outputDim=20,isRecurrent=True,isShortCircuit=True)
    layer[5]=aNNetwork.layer(layer=tpLayer.link,outputDim=20)
    layer[6]=aNNetwork.layer(layer=tpLayer.link,outputDim=20,isRecurrent=True)
    layer[7]=aNNetwork.layer(layer=tpLayer.link,outputDim=10,actFunc=actFunc.norm)
    rnn=aNNetwork.RNN_Bi(layer,numOfMaxRecurrent=4)
    aNNetwork.trainByAdamStep(rnn,trainDS,testDS,10,50000)

main()
'''
