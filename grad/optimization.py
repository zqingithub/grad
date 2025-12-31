from __future__ import annotations
from ast import List
import torch
import random
import multiprocessing
from artificial_neural_network import ANN, typeOfLayer
from auxiliary_tools import computeAcc, dataSet, showTrainProcess
from gradMatrix import gradMatrix
import math

def trainBySameStep(aNN:ANN,trainDS:dataSet,testDS:dataSet,step:float,gradClip:float=10,round:float=1.2,preIndex:float=0.9,minRound:int=2,showXSiez:int=0):

    nModel=len(aNN.modelPara)
    iteNum=math.ceil(round*trainDS.x.shape[0]/aNN.batchSize)
    gradPre:List[torch.tensor]=[]

    if showXSiez==0:
        showXSiez=200

    if showXSiez>iteNum:
        showXSiez=iteNum

    queueOfAcc=multiprocessing.Queue(maxsize=10)
    showProcess=multiprocessing.Process(target=showTrainProcess,name='showTrainProcess',kwargs={'size':showXSiez,'queue':queueOfAcc,'modelName':aNN.modelName},daemon=True)
    showProcess.start()

    for i in range(nModel):
        gradPre.append(torch.zeros(aNN.modelPara[i].grad.shape,dtype=torch.float))

    showN=1

    for i in range(iteNum):       
        for m in range(minRound):
            aNN.setXBatch(trainDS.x,i*aNN.batchSize%trainDS.x.shape[0])
            aNN.setRealYBatch(trainDS.y,i*aNN.batchSize%trainDS.x.shape[0])
            aNN.computeValue()
            aNN.computeGrad()

            for k in range(nModel):
                aNN.modelPara[k].grad[aNN.modelPara[k].grad>gradClip]=gradClip
                aNN.modelPara[k].grad[aNN.modelPara[k].grad<-gradClip]=-gradClip
                gradPre[k]=gradPre[k]*preIndex+(1-preIndex)*step*aNN.modelPara[k].grad
                aNN.modelPara[k].value=aNN.modelPara[k].value-gradPre[k]

            print("round:",i,":",m,"lossValue:",aNN.getLossValue())
        
        if (i+1)*1.0/iteNum*showXSiez>showN:
            showN+=1
            acc=computeAcc(aNN,testDS)
            print("acc:",acc*100,"%")
            queueOfAcc.put(acc)

        '''
        if i%1000==0:
            pyplot.figure(figsize=(8,8))
            pyplot.title(aNN.realY.value.cpu())
            pyplot.imshow(aNN.x.value.reshape(28,28).cpu())
            pyplot.show()
        '''

    aNN.createFinishModel()
    aNN.updateParaToFinishModel()
    print("testDate_acc:",computeAcc(aNN,testDS,testDS.x.shape[0],True)*100,"%")
    if aNN.batchSize>1:
        isHaveBatchNormalization=False
        for i in aNN.layerInfo:
            if i.typeOfLayer==typeOfLayer.batchNormalization:
                isHaveBatchNormalization=True
        if isHaveBatchNormalization:
            aNN.updateParaToFinishModel(isUseTrainData=True,trainData=trainDS.x)
            print("testDate_acc(use train data to compute bnPara):",computeAcc(aNN,testDS,testDS.x.shape[0],True)*100,"%")
    input("please input enter key to end the train")
    queueOfAcc.put(-1)
    showProcess.terminate()


def rollModelPara(para:List[gradMatrix]):
    for ite in para:
        tempValue=ite.value.view(-1)
        for i in range(tempValue.shape[0]):
            pos=random.randint(i,tempValue.shape[0]-1)
            ex=tempValue[pos].item()
            tempValue[pos]=tempValue[i]
            tempValue[i]=ex



def trainByAdamStep(aNN:ANN,trainDS:dataSet,testDS:dataSet,round:float=1.2,step:float=0.01,gradClip:float=10,\
                    beta1:float=0.9,beta2:float=0.999,esp:float=1e-8,minRound:int=2,showXSiez:int=0):

    nModel=len(aNN.modelPara)
    iteNum=math.ceil(round*trainDS.x.shape[0]/aNN.batchSize)
    gradPre:List[torch.tensor]=[]
    gradTotal:List[torch.tensor]=[]
    adjust1:float=beta1
    adjust2:float=beta2

    if showXSiez==0:
        showXSiez=200

    if showXSiez>iteNum:
        showXSiez=iteNum

    queueOfAcc=multiprocessing.Queue(maxsize=10)
    showProcess=multiprocessing.Process(target=showTrainProcess,name='showTrainProcess',kwargs={'size':showXSiez,'queue':queueOfAcc,'modelName':aNN.modelName},daemon=True)
    showProcess.start()

    for i in range(nModel):
        gradPre.append(torch.zeros(aNN.modelPara[i].grad.shape,dtype=torch.float))
        gradTotal.append(torch.zeros(aNN.modelPara[i].grad.shape,dtype=torch.float))

    showN=1

    for i in range(iteNum):
        for m in range(minRound):
            aNN.setXBatch(trainDS.x,i*aNN.batchSize%trainDS.x.shape[0])
            aNN.setRealYBatch(trainDS.y,i*aNN.batchSize%trainDS.x.shape[0])
            aNN.computeValue()
            aNN.computeGrad()
            for k in range(nModel):
                aNN.modelPara[k].grad[aNN.modelPara[k].grad>gradClip]=gradClip
                aNN.modelPara[k].grad[aNN.modelPara[k].grad<-gradClip]=-gradClip
                gradPre[k]=beta1*gradPre[k]+(1-beta1)*aNN.modelPara[k].grad
                gradTotal[k]=beta2*gradTotal[k]+(1-beta2)*aNN.modelPara[k].grad*aNN.modelPara[k].grad
                aNN.modelPara[k].value=aNN.modelPara[k].value-gradPre[k]*(step/(1-adjust1))/(torch.sqrt(gradTotal[k]*(1/(1-adjust2))+esp))
            adjust1*=beta1
            adjust2*=beta2

            print("round:",i,":",m,"lossValue:",aNN.getLossValue())
       

        if (i+1)*1.0/iteNum*showXSiez>showN:
            showN+=1
            acc=computeAcc(aNN,testDS)
            print("acc:",acc*100,"%")
            queueOfAcc.put(acc)

        '''
        if i%1000==0:
            pyplot.figure(figsize=(8,8))
            pyplot.title(aNN.realY.value.cpu())
            pyplot.imshow(aNN.x.value.reshape(28,28).cpu())
            pyplot.show()
        '''

    aNN.createFinishModel()
    aNN.updateParaToFinishModel()
    print("testDate_acc:",computeAcc(aNN,testDS,testDS.x.shape[0],True)*100,"%")
    if aNN.batchSize>1:
        isHaveBatchNormalization=False
        for i in aNN.layerInfo:
            if i.typeOfLayer==typeOfLayer.batchNormalization:
                isHaveBatchNormalization=True
        if isHaveBatchNormalization:
            aNN.updateParaToFinishModel(isUseTrainData=True,trainData=trainDS.x)
            print("testDate_acc(use train data to compute bnPara):",computeAcc(aNN,testDS,testDS.x.shape[0],True)*100,"%")
    input("please input enter key to end the train")
    queueOfAcc.put(-1)
    showProcess.terminate()



