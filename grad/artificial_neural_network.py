from __future__ import annotations
from ast import Dict, List, Set
import torch
from gradMatrix import gradMatrix as GM
from enum import Enum
import math
import gradMatrix


class actFunc(Enum):
    sigmoid=0
    Non=1
    exp=2
    ReLU=3
    norm=4
    softMax=5
    LReLU=6

class lossFunc(Enum):
    L1=0
    L2=1
    L_infin=2
    crossEntropy=3

class typeOfLayer(Enum):
    input=0
    conv=1
    reshape=2
    pool=3
    batchNormalization=4
    output=5

class poolMethod(Enum):
    maxPool=0
    avePool=1

class recurrentLinkMode(Enum):
    add=0
    concatenate=1
    addCon=2

class recurrentTimeWeightModel(Enum):
    allOne=0
    average=1
    classifier=2

class layerInfo:
    def __init__(self,channelSize:List[int]=None,layerType:typeOfLayer=typeOfLayer.conv,kernelSize:List[List[int]]=None,aFunc:List[actFunc]=None,pMethod:List[poolMethod]=None,\
        channelTable:List[List[int]]=None,shortCircuitAddSet:Set[int]={},shortCircuitConcateSet:Set[int]={},isRecurrent:bool=False,rcLinkMode:recurrentLinkMode=recurrentLinkMode.concatenate):
        self.typeOfLayer:typeOfLayer=layerType
        self.kernelSize:List[List[int]]=kernelSize
        self.channelSize:List[int]=channelSize
        self.actFunc:List[actFunc]=aFunc
        self.poolMethod:List[poolMethod]=pMethod
        self.channelTable:List[List[int]]=channelTable
        self.shortCircuitAddSet:Set[int]=shortCircuitAddSet
        self.shortCircuitConcateSet:Set[int]=shortCircuitConcateSet
        self.isRecurrent:bool=isRecurrent
        self.recurrentLinkMode:recurrentLinkMode=rcLinkMode

        if self.typeOfLayer==typeOfLayer.conv:
            if self.kernelSize==None:
                self.kernelSize=[[1,1]]*len(self.channelSize)
            if self.actFunc==None:
                self.actFunc=[actFunc.LReLU]*len(self.kernelSize)
            return

        if self.typeOfLayer==typeOfLayer.pool:
            if self.poolMethod==None:
                self.poolMethod=[poolMethod.avePool]
            if self.actFunc==None:
                self.actFunc=[actFunc.Non]*len(self.poolMethod)
            return

        if self.typeOfLayer==typeOfLayer.batchNormalization:
            if self.actFunc==None:
                self.actFunc=[actFunc.Non]
            return
        


def sigmoid(input:GM)->GM:
    return input.sigmoid()

def exp(input:GM)->GM:
    return input.exp()

def ReLU(input:GM)->GM:
    return input.ReLU()

def LReLU(input:GM)->GM:
    return input.LReLU()

def norm(input:GM)->GM:
    tempShape=input.value.shape
    input=input.abs().reshape([1,math.prod(input.value.shape)])
    tempSum=input@GM(torch.ones(input.value.shape,dtype=torch.float).T)
    input=input/tempSum.repeat(input.value.shape)
    return input.reshape(tempShape)


def softMax(input:GM)->GM:
    return input.softmax()

def L1(Y:GM,realY:GM)->GM:
    temp=(Y-realY).abs()
    return temp@GM(torch.ones(temp.value.shape,dtype=torch.float).T)

def L2(Y:GM,realY:GM)->GM:
    return (Y-realY)@((Y-realY).transpose())

def L_infin(Y:GM,realY:GM)->GM:
    return (Y-realY).abs().max()

def crossEntropy(Y:GM,realY:GM)->GM:
    temp=Y+GM(torch.ones(Y.value.shape,dtype=torch.float)*1e-37)
    return (realY*GM(torch.ones(realY.value.shape,dtype=torch.float)*(-1)))@((temp.log()).transpose())

def doActivation(input:GM,actFun:actFunc)->GM:
    if actFun==actFunc.sigmoid:
        input=sigmoid(input)
    if actFun==actFunc.exp:
        input=exp(input)
    if actFun==actFunc.ReLU:
        input=ReLU(input)
    if actFun==actFunc.norm:
        input=norm(input)
    if actFun==actFunc.LReLU:
        input=LReLU(input)
    if actFun==actFunc.softMax:
        input=softMax(input)
    return input

def doPoolMethod(input:GM,poolM:poolMethod,kernelSize:int)->GM:
    if poolM==poolMethod.avePool:
        return input.avgPool2D(kernelSize)
    if poolM==poolMethod.maxPool:
        return input.maxPool2D(kernelSize)


def doLossFunc(Y:GM,realY:GM,lsFunc:lossFunc)->GM:
    if lsFunc==lossFunc.L1:
        return L1(Y,realY)
    if lsFunc==lossFunc.L2:
        return L2(Y,realY)
    if lsFunc==lossFunc.L_infin:
        return L_infin(Y,realY)
    if lsFunc==lossFunc.crossEntropy:
        return crossEntropy(Y,realY)
    return None

        
#人工神经网络 Artificial Neural Network
class ANN:
    def __init__(self,modelName:str="ANN",batchSize:int=1,numOfMaxRecurrent:int=1,\
        recurrentTimeWeightModel:recurrentTimeWeightModel=recurrentTimeWeightModel.allOne,isBidirectional:bool=False):
        self.modelName:str=modelName
        self.batchSize:int=batchSize
        self.x:List[GM]=[None]*batchSize
        self.y:List[GM]=[None]*batchSize
        self.realY:List[GM]=[None]*batchSize
        self.lossValue:GM=None
        self.allLossValue:float=0.0
        self.modelPara:List[GM]=None
        self.modelParaGradStore:List[torch.tensor]=None
        self.layerInfo:List[layerInfo]=None
        self.layerOutput:List[List[GM]]=[None]*batchSize
        self.layerInput:List[GM]=[None]*batchSize
        self.lossFunc:lossFunc=None
        self.scDictAdd:Dict[int,List]={}
        self.scDictCon:Dict[int,List]={}
        self.bnMeanValue:List[GM]=None
        self.tempBnMV:GM=None
        self.bnDeviation:List[GM]=None
        self.tempBnSD:GM=None
        self.bnMvEma:List[torch.tensor]=None
        self.bnDeEma:List[torch.tensor]=None
        self.bnMvEmaPerTime:List[List[torch.tensor]]=[None]*numOfMaxRecurrent
        self.bnDeEmaPerTime:List[List[torch.tensor]]=[None]*numOfMaxRecurrent
        self.finishModel:ANN=None
        self.preIndexBn:float=0.99
        self.poolModelPara:List[List[GM]]=None
        self.posOfShortCircuitPara:int=0
        self.numOfMaxRecurrent:int=numOfMaxRecurrent
        self.nowRecurrent:int=0
        self.recurrentTimeWeightModel:recurrentTimeWeightModel=recurrentTimeWeightModel
        self.__recurrentWeight:List[float]=None
        self.isBidirectional:bool=isBidirectional
        self.XRecord:List[List[torch.tensor]]=[None]*self.numOfMaxRecurrent
        self.YRecord:List[List[torch.tensor]]=[None]*self.numOfMaxRecurrent
        self.RealYRecord:List[List[torch.tensor]]=[None]*self.numOfMaxRecurrent
        for i in range(self.numOfMaxRecurrent):
            self.XRecord[i]=[None]*batchSize
            self.YRecord[i]=[None]*batchSize
            self.RealYRecord[i]=[None]*batchSize
        self.recurrentLeftInput:List[List[GM]]=[None]*batchSize
        self.recurrentRightInput:List[List[GM]]=[None]*batchSize
        self.recurrentInputPreGrad:List[List[torch.tensor]]=[None]*batchSize
        self.recurrentInput:List[List[GM]]=[None]*batchSize
        self.recurrentOutputLayerNo:List[int]=None
        self.rcLeftInputStore:List[List[List[torch.tensor]]]=[None]*(numOfMaxRecurrent+1)
        self.rcRightInputStore:List[List[List[torch.tensor]]]=[None]*(numOfMaxRecurrent+1)
        self.rcLeftInputGradStore:List[List[List[torch.tensor]]]=[None]*(numOfMaxRecurrent+1)
        self.rcRightInputGradStore:List[List[List[torch.tensor]]]=[None]*(numOfMaxRecurrent+1)
        for i in range(numOfMaxRecurrent+1):
            self.rcLeftInputStore[i]=[None]*batchSize
            self.rcRightInputStore[i]=[None]*batchSize
            self.rcLeftInputGradStore[i]=[None]*batchSize
            self.rcRightInputGradStore[i]=[None]*batchSize
        self.yClassifier:List[torch.tensor]=[None]*batchSize

    def __createModelPara(self,layerNo):
        input:torch.tensor=self.layerInput[0].value
        layInfo:layerInfo=self.layerInfo[layerNo]

        if layInfo.typeOfLayer==typeOfLayer.batchNormalization:
            self.modelPara.append(GM(torch.rand(input.shape,dtype=torch.float)/2,gradMatrix.variableType.gradVariable))
            self.modelPara.append(GM(torch.rand(input.shape,dtype=torch.float)/2,gradMatrix.variableType.gradVariable))
            return

        if layInfo.typeOfLayer==typeOfLayer.conv:
            inSum:List[int]=[]
            tempCT=layInfo.channelTable
            if tempCT==None:
                tempCT=[list(range(input.shape[1]))]*len(layInfo.kernelSize)
            if layInfo.channelTable==None:
                inSum=[input.shape[1]]*len(layInfo.kernelSize)
            else:
                for i in layInfo.channelTable:
                    inSum.append(len(i))
            for i in range(len(layInfo.kernelSize)):
                temp=torch.rand([layInfo.channelSize[i],inSum[i]+1,layInfo.kernelSize[i][0],layInfo.kernelSize[i][1]],dtype=torch.float)
                temp=temp/(math.prod(temp.shape)/layInfo.channelSize[i])
                self.modelPara.append(GM(temp,gradMatrix.variableType.gradVariable))
            return

        if layInfo.typeOfLayer==typeOfLayer.pool:
            inSum:List[int]=[]
            if layInfo.channelTable==None:
                inSum=[input.shape[1]]*len(layInfo.actFunc)
            else:
                for i in layInfo.channelTable:
                    inSum.append(len(i))
            for i in range(len(layInfo.actFunc)):
                if layInfo.actFunc[i]!=actFunc.Non:
                    self.modelPara.append(GM(torch.rand([1,inSum[i],1,1],dtype=torch.float)/2,gradMatrix.variableType.gradVariable))
                    self.modelPara.append(GM(torch.rand([1,inSum[i],1,1],dtype=torch.float)/2,gradMatrix.variableType.gradVariable))
            return
              

    def __registerShortCircuit(self,layerNo:int,batchNo:int):
        if batchNo>0:
            return
        for i in self.layerInfo[layerNo].shortCircuitAddSet:
            index=i+layerNo
            if self.layerInfo[index].typeOfLayer==typeOfLayer.output:
                raise TypeError("short circuit must not link output layer")
            if i>1:
                self.scDictAdd[index].append(layerNo)
            else:
                raise TypeError("short circuit must up more 1 layer")
        for i in self.layerInfo[layerNo].shortCircuitConcateSet:
            index=i+layerNo
            if self.layerInfo[index].typeOfLayer==typeOfLayer.output:
                raise TypeError("short circuit must not link output layer")
            if i>1:
                self.scDictCon[index].append(layerNo)
            else:
                raise TypeError("short circuit must up more 1 layer")

    def __createShortCircuitPara(self,layerNoIn:int,layerNoTa:int)->int:
        if layerNoIn>=0:
            input:GM=self.layerOutput[0][layerNoIn]
        else:
            input:GM=self.recurrentInput[0][-layerNoIn-1]
        target:GM=self.layerOutput[0][layerNoTa]
        if input.value.shape[1]==target.value.shape[1]:
            return 0
        temp=torch.rand([target.value.shape[1],input.value.shape[1]+1,1,1],dtype=torch.float)/(input.value.shape[1]+1)
        self.modelPara.append(GM(temp,gradMatrix.variableType.gradVariable))
        return 1

    def __reshapeShortCircuitInput(self,layerNoIn:int,layerNoTa:int,batchNo:int)->GM:
        if layerNoIn>=0:
            input:GM=self.layerOutput[batchNo][layerNoIn]
        else:
            input:GM=self.recurrentInput[batchNo][-layerNoIn-1]
        target:GM=self.layerOutput[batchNo][layerNoTa]
        if input.value.shape[2]==target.value.shape[2] and input.value.shape[3]==target.value.shape[3]:
            return input
        if target.value.shape[2]==1 and target.value.shape[3]==1:
            return input.reshape([1,math.prod(input.value.shape),1,1])
        index=1
        for i in range(layerNoIn+1,layerNoTa+1):
            if self.layerInfo[i].typeOfLayer==typeOfLayer.pool:
                index=index*self.layerInfo[i].kernelSize[0][0]
        return input.avgPool2D(index)

    def __reshapeShortCiruitChannel(self,input:GM,kernel:GM=None)->GM:
        temp=GM(torch.ones([1,1,input.value.shape[2],input.value.shape[3]],dtype=torch.float))
        input=GM.cat(input,temp,1)
        return input.conv2D(kernel).LReLU()

    def __linkShortCircuit(self,layerNo:int,batchNo:int):
        if layerNo<0:
            return
        target:GM=self.layerOutput[batchNo][layerNo]
        outcome:GM=target
        if len(self.scDictAdd[layerNo])>0:
            input:List[GM]=[]
            for ly in self.scDictAdd[layerNo]: 
                input.append(self.__reshapeShortCircuitInput(ly,layerNo,batchNo))
            if batchNo==0:
                self.posOfShortCircuitPara=0
                for ly in self.scDictAdd[layerNo]:
                    self.posOfShortCircuitPara+=self.__createShortCircuitPara(ly,layerNo)
                self.posOfShortCircuitPara=len(self.modelPara)-self.posOfShortCircuitPara
            index=0
            for ly in input:
                if ly.value.shape[1]==target.value.shape[1]:
                    outcome=outcome+ly
                else:
                    outcome=outcome+self.__reshapeShortCiruitChannel(ly,self.modelPara[self.posOfShortCircuitPara+index])
                    index+=1
            outcome=outcome/GM(torch.ones(outcome.value.shape,dtype=torch.float)*len(input))

        if  len(self.scDictCon[layerNo])>0:
            for ly in self.scDictCon[layerNo]:
                outcome=GM.cat(outcome,self.__reshapeShortCircuitInput(ly,layerNo,batchNo),1)
        self.layerInput[batchNo]=outcome

        
    def __createInputLayer(self,layerNo:int,batchNo:int):
        if layerNo!=0:
            raise TypeError("input layer must in first layer")
        layInfo:layerInfo=self.layerInfo[0]
        self.x[batchNo]=GM(torch.zeros([1,layInfo.channelSize[0],layInfo.kernelSize[0][0],layInfo.kernelSize[0][1]],dtype=torch.float))
        self.layerOutput[batchNo][0]=self.x[batchNo]
        self.layerInput[batchNo]=self.x[batchNo]

    def __createOutputLayer(self,layerNo:int,batchNo:int):
        temp:GM=self.layerInput[batchNo]
        self.layerOutput[batchNo][layerNo]=temp.reshape([1,math.prod(temp.value.shape)])

    def __createReshapeLayer(self,layerNo:int,batchNo:int):
        layInfo:layerInfo=self.layerInfo[layerNo]
        input:GM=self.layerInput[batchNo]
        channelSize=math.prod(input.value.shape)//layInfo.kernelSize[0][0]//layInfo.kernelSize[0][1]
        temp=input.reshape([1,channelSize,layInfo.kernelSize[0][0],layInfo.kernelSize[0][1]])
        self.layerOutput[batchNo][layerNo]=temp

    def __createConvLayer(self,layerNo:int,batchNo:int):
        outputStore:List[GM]=[]
        layInfo:layerInfo=self.layerInfo[layerNo]
        input=GM(torch.ones((1,1,self.layerInput[batchNo].value.shape[2],self.layerInput[batchNo].value.shape[3]),dtype=torch.float))
        input=GM.cat(self.layerInput[batchNo],input,1)
        if batchNo==0:
            self.__createModelPara(layerNo)
        kSum=len(layInfo.kernelSize)
        for i in range(kSum):
            if layInfo.channelTable==None:
                outputStore.append(input.conv2D(self.modelPara[i-kSum]))
            else:
                index=torch.tensor(layInfo.channelTable[i]+[input.value.shape[1]-1])
                outputStore.append(input.index_select(1,index).conv2D(self.modelPara[i-kSum]))
            outputStore[-1]=doActivation(outputStore[-1],layInfo.actFunc[i])
        tempOutput=outputStore[0]
        for i in range(1,len(outputStore)):
            tempOutput=GM.cat(tempOutput,outputStore[i],1)
        self.layerOutput[batchNo][layerNo]=tempOutput

    def __createPoolLayer(self,layerNo:int,batchNo:int):
        outputStore:List[GM]=[]
        layInfo:layerInfo=self.layerInfo[layerNo]
        input:GM=self.layerInput[batchNo]
        if batchNo==0:
            self.__createModelPara(layerNo)
            h=input.value.shape[2]//layInfo.kernelSize[0][0]
            w=input.value.shape[3]//layInfo.kernelSize[0][0]
            self.poolModelPara=[]
            self.poolModelPara.append([])
            self.poolModelPara.append([])
            c=0
            for i in range(len(layInfo.actFunc)):
                if layInfo.actFunc[i]!=actFunc.Non:
                    c=c+1
            for i in range(c,0,-1):
                self.poolModelPara[0].append(self.modelPara[-i*2].repeat([1,1,h,w]))
                self.poolModelPara[1].append(self.modelPara[-i*2+1].repeat([1,1,h,w]))
        pSum=len(layInfo.poolMethod)
        actCount=0
        for i in range(pSum):
            if layInfo.channelTable==None:
                outputStore.append(doPoolMethod(input,layInfo.poolMethod[i],layInfo.kernelSize[0][0]))
            else:
                index=torch.tensor(layInfo.channelTable[i])
                outputStore.append(doPoolMethod(input.index_select(1,index),layInfo.poolMethod[i],layInfo.kernelSize[0][0]))
            if layInfo.actFunc[i]!=actFunc.Non:
                outputStore[-1]=outputStore[-1]*self.poolModelPara[0][actCount]+self.poolModelPara[1][actCount]
                actCount=actCount+1
                outputStore[-1]=doActivation(outputStore[-1],layInfo.actFunc[i])
        tempOutput=outputStore[0]
        for i in range(1,len(outputStore)):
            tempOutput=GM.cat(tempOutput,outputStore[i],1)
        self.layerOutput[batchNo][layerNo]=tempOutput


    def __createBatchNormalization(self,layerNo:int,batchNo:int):
        if self.batchSize<2:
            input:GM=self.layerInput[batchNo]
            self.bnMeanValue.append(GM(torch.zeros(input.value.shape,dtype=torch.float)))
            self.bnDeviation.append(GM(torch.ones(input.value.shape,dtype=torch.float)))
            temp=(self.bnDeviation[-1]+GM(torch.ones(input.value.shape,dtype=torch.float)*1e-15)).pow(0.5)+GM(torch.ones(input.value.shape,dtype=torch.float)*1e-15)
            if batchNo==0:
                self.__createModelPara(layerNo)       
            temp=(input-self.bnMeanValue[-1])/temp
            temp=temp*self.modelPara[-2]+self.modelPara[-1]
            self.layerOutput[0][layerNo]=doActivation(temp,self.layerInfo[layerNo].actFunc[0])
        else:
            if batchNo==0:
                self.tempBnMV=GM(torch.zeros(self.layerInput[batchNo].value.shape,dtype=torch.float))
                for i in range(self.batchSize):
                    self.tempBnMV=self.tempBnMV+self.layerInput[i]
                self.tempBnMV=self.tempBnMV/GM(torch.ones(self.tempBnMV.value.shape,dtype=torch.float)*self.batchSize)
                self.bnMeanValue.append(self.tempBnMV)
                self.tempBnSD=GM(torch.zeros(self.layerInput[batchNo].value.shape,dtype=torch.float))
                for i in range(self.batchSize):
                    temp=(self.layerInput[i]-self.tempBnMV)*(self.layerInput[i]-self.tempBnMV)
                    self.tempBnSD=self.tempBnSD+temp
                self.tempBnSD=self.tempBnSD/GM(torch.ones(self.tempBnMV.value.shape,dtype=torch.float)*self.batchSize)
                self.bnDeviation.append(self.tempBnSD)
                self.tempBnSD=(self.tempBnSD+GM(torch.ones(self.tempBnMV.value.shape,dtype=torch.float)*1e-15)).pow(0.5)+GM(torch.ones(self.tempBnMV.value.shape,dtype=torch.float)*1e-15)
                self.__createModelPara(layerNo)
        
            temp=(self.layerInput[batchNo]-self.tempBnMV)/self.tempBnSD
            temp=temp*self.modelPara[-2]+self.modelPara[-1]
            self.layerOutput[batchNo][layerNo]=doActivation(temp,self.layerInfo[layerNo].actFunc[0])
        


    def __createLayer(self,layerNo:int,batchNo:int):
        if self.layerInfo[layerNo].typeOfLayer==typeOfLayer.input:
            self.__createInputLayer(layerNo,batchNo)
            return
        if self.layerInfo[layerNo].typeOfLayer==typeOfLayer.reshape:
            self.__createReshapeLayer(layerNo,batchNo)
            return
        if self.layerInfo[layerNo].typeOfLayer==typeOfLayer.batchNormalization:
            self.__createBatchNormalization(layerNo,batchNo)
            return
        if self.layerInfo[layerNo].typeOfLayer==typeOfLayer.conv:
            self.__createConvLayer(layerNo,batchNo)
            return
        if self.layerInfo[layerNo].typeOfLayer==typeOfLayer.pool:
            self.__createPoolLayer(layerNo,batchNo)
            return
        if self.layerInfo[layerNo].typeOfLayer==typeOfLayer.output:
            self.__createOutputLayer(layerNo,batchNo)
            return

    def __createRecurrent(self,layerNo:int,batchNo:int):
        if layerNo<1 or layerNo>=len(self.layerInfo):
            return
        layInfo:layerInfo=self.layerInfo[layerNo]
        if layInfo.isRecurrent==False:
            return
        if layInfo.typeOfLayer!=typeOfLayer.conv:
            raise TypeError("recurrent layer must is conv layer")
        outputLen=0
        for i in range(len(layInfo.channelSize)):
            outputLen+=layInfo.channelSize[i]
        input:torch.tensor=self.layerOutput[0][layerNo-1].value
        self.recurrentLeftInput[batchNo].append(GM(torch.rand([1,outputLen,input.shape[2],input.shape[3]],dtype=torch.float),gradMatrix.variableType.gradVariable))
        self.recurrentRightInput[batchNo].append(GM(torch.rand([1,outputLen,input.shape[2],input.shape[3]],dtype=torch.float),gradMatrix.variableType.gradVariable))
        self.recurrentInput[batchNo].append(self.recurrentLeftInput[batchNo][-1])
        if self.isBidirectional:
            self.recurrentInput[batchNo].append(self.recurrentRightInput[batchNo][-1])
        if batchNo==0:
            reInSize=len(self.recurrentInput[0])
            if layInfo.recurrentLinkMode==recurrentLinkMode.concatenate or layInfo.recurrentLinkMode==recurrentLinkMode.addCon:
                self.scDictCon[layerNo-1].append(-reInSize)
                if self.isBidirectional:
                    self.scDictCon[layerNo-1].append(-reInSize+1)
            if layInfo.recurrentLinkMode==recurrentLinkMode.add or layInfo.recurrentLinkMode==recurrentLinkMode.addCon:
                self.scDictAdd[layerNo-1].append(-reInSize)
                if self.isBidirectional:
                    self.scDictAdd[layerNo-1].append(-reInSize+1)



    def createAnn(self,lyInfo:List[layerInfo],lsFunc:lossFunc=lossFunc.L2,isCheckShare:bool=True):
        self.modelPara=[]
        self.modelParaGradStore=[]
        self.layerInfo=lyInfo
        self.lossFunc=lsFunc
        for i in range(self.batchSize):
            self.layerOutput[i]=[None]*len(lyInfo)
            self.recurrentLeftInput[i]=[]
            self.recurrentRightInput[i]=[]
            self.recurrentInput[i]=[]
            self.recurrentInputPreGrad[i]=[]
        self.layerInput=[None]*self.batchSize
        self.scDictAdd={}
        self.scDictCon={}
        for i in range(len(self.layerInfo)):
            self.scDictAdd[i]=[]
            self.scDictCon[i]=[]
        self.bnMeanValue=[]
        self.bnDeviation=[]
        self.bnMvEma=[]
        self.bnDeEma=[]


        for i in range(len(lyInfo)):
            for j in range(self.batchSize):
                self.__createRecurrent(i,j)
            for j in range(self.batchSize):
                self.__linkShortCircuit(i-1,j)
            for j in range(self.batchSize):
                self.__createLayer(i,j)
            for j in range(self.batchSize):
                self.__registerShortCircuit(i,j)

        self.lossValue=GM(torch.zeros((1,1),dtype=torch.float))
        for i in range(self.batchSize):
            self.x[i]=self.layerOutput[i][0]
            self.y[i]=self.layerOutput[i][-1]
            self.realY[i]=GM(torch.zeros(self.y[0].value.shape,dtype=torch.float))
            self.lossValue=self.lossValue+doLossFunc(self.y[i],self.realY[i],lsFunc)
        self.lossValue=self.lossValue/GM(torch.ones((1,1),dtype=torch.float)*self.batchSize)

        for i in range(self.numOfMaxRecurrent):
            self.bnMvEmaPerTime[i]=[]
            self.bnDeEmaPerTime[i]=[]

        for i in range(len(self.modelPara)):
            self.modelParaGradStore.append(torch.zeros(self.modelPara[i].grad.shape,dtype=torch.float))

        rcInputLen=len(self.recurrentLeftInput[0])
        for i in range(self.numOfMaxRecurrent+1):          
            for j in range(self.batchSize):
                self.rcLeftInputStore[i][j]=[None]*rcInputLen
                self.rcRightInputStore[i][j]=[None]*rcInputLen
                self.rcLeftInputGradStore[i][j]=[None]*rcInputLen
                self.rcRightInputGradStore[i][j]=[None]*rcInputLen

        self.recurrentOutputLayerNo=[]
        for i in range(len(self.layerInfo)):
            if self.layerInfo[i].isRecurrent:
                self.recurrentOutputLayerNo.append(i)

        for i in range(self.batchSize):
            for j in range(len(self.recurrentLeftInput[0])):
                self.recurrentInputPreGrad[i].append(torch.zeros(self.recurrentLeftInput[0][j].grad.shape,dtype=torch.float))

        for i in range(len(self.bnMeanValue)):
            self.bnMvEma.append(torch.zeros(self.bnMeanValue[i].value.shape,dtype=torch.float))
            self.bnDeEma.append(torch.zeros(self.bnDeviation[i].value.shape,dtype=torch.float))
            for j in range(self.numOfMaxRecurrent):
                self.bnMvEmaPerTime[j].append(torch.zeros(self.bnMeanValue[i].value.shape,dtype=torch.float))
                self.bnDeEmaPerTime[j].append(torch.zeros(self.bnDeviation[i].value.shape,dtype=torch.float))

        if isCheckShare:
            self.lossValue.checkShare()


    def createFinishModel(self):
         self.finishModel=ANN(modelName=self.modelName+'_finishModel',batchSize=1,numOfMaxRecurrent=self.numOfMaxRecurrent,isBidirectional=self.isBidirectional)
         self.finishModel.createAnn(lyInfo=self.layerInfo,lsFunc=self.lossFunc)

    def updateParaToFinishModel(self,isUseTrainData:bool=False,trainData:torch.tensor=None):
        for i in range(len(self.modelPara)):
            self.finishModel.modelPara[i].value=self.modelPara[i].value.clone()
        if isUseTrainData:
            iter=math.ceil(trainData.shape[0]*1.0/self.batchSize)
            for i in range(len(self.finishModel.bnMeanValue)):
                self.finishModel.bnMvEma[i]=self.bnMvEma[i].clone()
                self.bnMvEma[i].zero_()
                self.finishModel.bnDeEma[i]=self.bnDeEma[i].clone()
                self.bnDeEma[i].zero_()
                for j in range(self.finishModel.numOfMaxRecurrent):
                    self.finishModel.bnMvEmaPerTime[j][i]=self.bnMvEmaPerTime[j][i].clone()
                    self.bnMvEmaPerTime[j][i].zero_()
                    self.finishModel.bnDeEmaPerTime[j][i]=self.bnDeEmaPerTime[j][i].clone()
                    self.bnDeEmaPerTime[j][i].zero_()
            for i in range(iter):
                self.setXBatch(trainData,iter*self.batchSize)
                self.computeValue()
            for i in range(len(self.finishModel.bnMeanValue)):
                temp=self.finishModel.bnMvEma[i]
                self.finishModel.bnMvEma[i]=self.bnMvEma[i]
                self.bnMvEma[i]=temp
                temp=self.finishModel.bnDeEma[i]
                self.finishModel.bnDeEma[i]=self.bnDeEma[i]
                self.bnDeEma[i]=temp
                for j in range(self.finishModel.numOfMaxRecurrent):
                    temp=self.finishModel.bnMvEmaPerTime[j][i]
                    self.finishModel.bnMvEmaPerTime[j][i]=self.bnMvEmaPerTime[j][i]
                    self.bnMvEmaPerTime[j][i]=temp
                    temp=self.finishModel.bnDeEmaPerTime[j][i]
                    self.finishModel.bnDeEmaPerTime[j][i]=self.bnDeEmaPerTime[j][i].clone()
                    self.bnDeEmaPerTime[j][i]=temp
        else:
            for i in range(len(self.bnMeanValue)):
                self.finishModel.bnMvEma[i]=self.bnMvEma[i].clone()
                self.finishModel.bnDeEma[i]=self.bnDeEma[i].clone()
                for j in range(self.finishModel.numOfMaxRecurrent):
                    self.finishModel.bnMvEmaPerTime[j][i]=self.bnMvEmaPerTime[j][i].clone()
                    self.finishModel.bnDeEmaPerTime[j][i]=self.bnDeEmaPerTime[j][i].clone()


    def __initRecurrentWeight(self):
        if self.recurrentTimeWeightModel==recurrentTimeWeightModel.allOne:
            self.__recurrentWeight=[1.0]*self.nowRecurrent
            return
        if self.recurrentTimeWeightModel==recurrentTimeWeightModel.average:
            temp=1.0/self.nowRecurrent
            self.__recurrentWeight=[temp]*self.nowRecurrent
            return
        if self.recurrentTimeWeightModel==recurrentTimeWeightModel.classifier:
            if self.isBidirectional:
                temp=1.0/self.nowRecurrent
                self.__recurrentWeight=[temp]*self.nowRecurrent
            else:
                if self.nowRecurrent==1:
                    self.__recurrentWeight=[1.0]*self.nowRecurrent
                else:
                    step=2.0/self.nowRecurrent/(self.nowRecurrent-1)
                    self.__recurrentWeight=[0]*self.nowRecurrent
                    sum=0
                    for i in range(1,self.nowRecurrent-1,1):
                        self.__recurrentWeight[i]=self.__recurrentWeight[i-1]+step
                        sum+=self.__recurrentWeight[i]
                    self.__recurrentWeight[-1]=1-sum
            return

    def setXBatch(self,value:torch.tensor,beginPos:int):
        if self.nowRecurrent!=value.shape[1]:
            self.nowRecurrent=value.shape[1]
            self.__zeroBorderRecurrentInput()
            self.__initRecurrentWeight()
        for i in range(self.batchSize):
            pos=(beginPos+i)%value.shape[0]
            for j in range(self.nowRecurrent):
                self.XRecord[j][i]=value[pos][j].clone()

    def setRealYBatch(self,value:torch.tensor,beginPos:int):
        if self.nowRecurrent!=value.shape[1]:
            self.nowRecurrent=value.shape[1]
            self.__zeroBorderRecurrentInput()
            self.__initRecurrentWeight()
        for i in range(self.batchSize):
            pos=(beginPos+i)%value.shape[0]
            for j in range(self.nowRecurrent):
                self.RealYRecord[j][i]=value[pos][j].clone()

    def __loadRecurrentInput(self,timeNo:int,isLeftInput:bool,isLoadZero:bool=False):
        if isLeftInput:
            for i in range(self.batchSize):
                for j in range(len(self.recurrentLeftInput[0])):
                    self.recurrentLeftInput[i][j].value=self.rcLeftInputStore[timeNo][i][j]
                    if isLoadZero:
                        self.recurrentLeftInput[i][j].value=self.rcLeftInputStore[0][i][j]
        else:
            for i in range(self.batchSize):
                for j in range(len(self.recurrentRightInput[0])):
                    self.recurrentRightInput[i][j].value=self.rcRightInputStore[timeNo][i][j]
                    if isLoadZero:
                        self.recurrentRightInput[i][j].value=self.rcLeftInputStore[0][i][j]

    def __saveRecurrentInput(self,timeNo:int,isLeftInput:bool):
        if isLeftInput:
            for i in range(self.batchSize):
                for j in range(len(self.recurrentLeftInput[0])):
                    self.rcLeftInputStore[timeNo][i][j]=self.layerOutput[i][self.recurrentOutputLayerNo[j]].value.clone()
        else:
            for i in range(self.batchSize):
                for j in range(len(self.recurrentRightInput[0])):
                    self.rcRightInputStore[timeNo][i][j]=self.layerOutput[i][self.recurrentOutputLayerNo[j]].value.clone()

    def __zeroBorderRecurrentInput(self):
        for i in range(self.batchSize):
            for j in range(len(self.recurrentLeftInput[0])):
                self.rcLeftInputStore[0][i][j]=torch.zeros(self.recurrentLeftInput[0][j].value.shape,dtype=torch.float)
                self.rcRightInputStore[self.nowRecurrent-1][i][j]=self.rcLeftInputStore[0][i][j]

    def __setXBatchPerTime(self,timeNo:int):
        for i in range(self.batchSize):
            self.x[i].value=self.XRecord[timeNo][i]

    def __setYRealBatchPerTime(self,timeNo:int):
        for i in range(self.batchSize):
            if self.RealYRecord[timeNo][i]!=None:
                self.realY[i].value=self.RealYRecord[timeNo][i]

    def __saveYBatchPerTime(self,timeNo:int):
        for i in range(self.batchSize):
            self.YRecord[timeNo][i]=self.y[i].value.clone()

    def __updateBnPara(self,timeNo:int):
        for i in range(len(self.bnMeanValue)):
            self.bnMvEma[i]=self.preIndexBn*self.bnMvEma[i]+(1-self.preIndexBn)*self.bnMeanValue[i].value
            self.bnDeEma[i]=self.preIndexBn*self.bnDeEma[i]+(1-self.preIndexBn)*self.bnDeviation[i].value
            self.bnMvEmaPerTime[timeNo][i]=self.preIndexBn*self.bnMvEmaPerTime[timeNo][i]+(1-self.preIndexBn)*self.bnMeanValue[i].value
            self.bnDeEmaPerTime[timeNo][i]=self.preIndexBn*self.bnDeEmaPerTime[timeNo][i]+(1-self.preIndexBn)*self.bnDeviation[i].value

    def __loadBnPara(self,timeNo:int=0,isUseBnParaPerTime:bool=True):
        if isUseBnParaPerTime:
            for i in range(len(self.bnMeanValue)):
                self.bnMeanValue[i].value=self.bnMvEmaPerTime[timeNo][i]
                self.bnDeviation[i].value=self.bnDeEmaPerTime[timeNo][i]
        else:
            for i in range(len(self.bnMeanValue)):
                self.bnMeanValue[i].value=self.bnMvEma[i]
                self.bnDeviation[i].value=self.bnDeEma[i]

    def computeValue(self,isUpdataBnPara:bool=True,isUseBnParePerTime:bool=True):
        self.allLossValue=0.0
        if self.batchSize==1 and (not isUseBnParePerTime):
            self.__loadBnPara(isUseBnParaPerTime=False)
        for i in range(self.nowRecurrent):
            self.__loadRecurrentInput(i,isLeftInput=True)
            if self.isBidirectional:
                self.__loadRecurrentInput(i,isLeftInput=False,isLoadZero=True)
            self.__setXBatchPerTime(i)
            self.__setYRealBatchPerTime(i)
            if self.batchSize==1 and isUseBnParePerTime:
                self.__loadBnPara(i)
            self.lossValue.computeValue()
            self.allLossValue+=self.lossValue.value[0,0].item()*self.__recurrentWeight[i]
            self.__saveRecurrentInput(i+1,isLeftInput=True)
            self.__saveYBatchPerTime(i)
            if isUpdataBnPara and (not self.isBidirectional):
                self.__updateBnPara(i)
        if self.isBidirectional:
            self.allLossValue=0.0
            for i in range(self.nowRecurrent-1,-1,-1):
                self.__loadRecurrentInput(i,isLeftInput=False)
                self.__loadRecurrentInput(i,isLeftInput=True,isLoadZero=True)
                self.__setXBatchPerTime(i)
                self.__setYRealBatchPerTime(i)
                if self.batchSize==1 and isUseBnParePerTime:
                    self.__loadBnPara(i)
                self.lossValue.computeValue()
                self.__saveRecurrentInput(i-1,isLeftInput=False)
            for i in range(self.nowRecurrent):
                self.__loadRecurrentInput(i,isLeftInput=True)
                self.__loadRecurrentInput(i,isLeftInput=False)
                self.__setXBatchPerTime(i)
                self.__setYRealBatchPerTime(i)
                if self.batchSize==1 and isUseBnParePerTime:
                    self.__loadBnPara(i)
                self.lossValue.computeValue()
                self.allLossValue+=self.lossValue.value[0,0].item()*self.__recurrentWeight[i]
                self.__saveYBatchPerTime(i)
                if isUpdataBnPara:
                    self.__updateBnPara(i)

    def __zeroRecurrentInputGrad(self):
        for i in range(self.batchSize):
            for j in self.recurrentLeftInput[i]:
                j.grad.zero_()
            for j in self.recurrentRightInput[i]:
                j.grad.zero_()

    def __zerorecurrentInputPreGrad(self):
        for i in range(self.batchSize):
            for j in self.recurrentInputPreGrad[i]:
                j.zero_()

    def __updaterecurrentInputPreGrad(self,timeNo:int,isLeftInput:bool):
        if isLeftInput:
            for i in range(self.batchSize):
                for j in range(len(self.recurrentLeftInput[0])):
                    self.recurrentInputPreGrad[i][j]=self.recurrentLeftInput[i][j].grad+self.rcLeftInputGradStore[timeNo][i][j]
        else:
            for i in range(self.batchSize):
                for j in range(len(self.recurrentRightInput[0])):
                    self.recurrentInputPreGrad[i][j]=self.recurrentRightInput[i][j].grad+self.rcRightInputGradStore[timeNo][i][j]

    def __loadRecurrentInputPreGrad(self):
        for i in range(self.batchSize):
            for j in range(len(self.recurrentLeftInput[0])):
                self.layerOutput[i][self.recurrentOutputLayerNo[j]].sumOfGrad=self.recurrentInputPreGrad[i][j].clone()

    def __saveToRecurrentInputGradStore(self,timeNo:int):
        for i in range(self.batchSize):
            for j in range(len(self.recurrentLeftInput[0])):
                self.rcLeftInputGradStore[timeNo][i][j]=self.recurrentLeftInput[i][j].grad.clone()
                self.rcRightInputGradStore[timeNo][i][j]=self.recurrentRightInput[i][j].grad.clone()
                    

    def __addGradToModelParaGradStore(self):
        for i in range(len(self.modelPara)):
            self.modelParaGradStore[i]+=self.modelPara[i].grad

    def __loadModelParaGradStoreToGrad(self):
        for i in range(len(self.modelPara)):
            self.modelPara[i].grad=self.modelParaGradStore[i].clone()

    def computeGrad(self):
        if self.nowRecurrent<2:
            self.lossValue.computeGrad(self.modelPara)
            return
        for i in self.modelParaGradStore:
            i.zero_()
        for i in range(self.nowRecurrent):
            self.__loadRecurrentInput(i,isLeftInput=True,isLoadZero=False)
            if self.isBidirectional:
                self.__loadRecurrentInput(i,isLeftInput=False,isLoadZero=False)
            self.__setXBatchPerTime(i)
            self.__setYRealBatchPerTime(i)
            self.lossValue.computeValue()
            self.__zeroRecurrentInputGrad()
            self.lossValue.computeGrad(self.modelPara,torch.ones(self.lossValue.value.shape,dtype=torch.float)*self.__recurrentWeight[i])
            self.__addGradToModelParaGradStore()
            self.__saveToRecurrentInputGradStore(i)
        self.__zerorecurrentInputPreGrad()
        for i in range(self.nowRecurrent-1,-1,-1):
            self.__loadRecurrentInput(i,isLeftInput=True,isLoadZero=False)
            self.__loadRecurrentInput(i,isLeftInput=False,isLoadZero=True)
            self.__setXBatchPerTime(i)
            self.__setYRealBatchPerTime(i)
            self.lossValue.computeValue()
            self.__zeroRecurrentInputGrad()
            self.__loadRecurrentInputPreGrad()
            self.lossValue.computeGrad(self.modelPara,torch.zeros(self.lossValue.value.shape,dtype=torch.float))
            self.__addGradToModelParaGradStore()
            self.__updaterecurrentInputPreGrad(i,isLeftInput=True)
        if self.isBidirectional:
            self.__zerorecurrentInputPreGrad()
            for i in range(self.nowRecurrent):
                self.__loadRecurrentInput(i,isLeftInput=True,isLoadZero=True)
                self.__loadRecurrentInput(i,isLeftInput=False,isLoadZero=False)
                self.__setXBatchPerTime(i)
                self.__setYRealBatchPerTime(i)
                self.lossValue.computeValue()
                self.__zeroRecurrentInputGrad()
                self.__loadRecurrentInputPreGrad()
                self.lossValue.computeGrad(self.modelPara,torch.zeros(self.lossValue.value.shape,dtype=torch.float))
                self.__addGradToModelParaGradStore()
                self.__updaterecurrentInputPreGrad(i,isLeftInput=False)
        self.__loadModelParaGradStoreToGrad()

    def getLossValue(self):
        return self.allLossValue

    def computeYClassifier(self):
        for i in range(self.batchSize):
            tempY=torch.zeros(self.y[0].value.shape,dtype=torch.float)
            for j in range(self.nowRecurrent):
                tempY=tempY+(self.YRecord[j][i]*self.__recurrentWeight[j])
            self.yClassifier[i]=tempY.clone()

    def getYClassifier(self,x:torch.tensor)->torch.tensor:
        tempX=torch.zeros(x[0].shape,dtype=torch.float)
        for i in range(self.nowRecurrent):
            tempX=tempX+(x[i]*self.__recurrentWeight[i])
        return tempX

    def computeGradByEstimate(self,step:float=0.01):
        for g in self.modelPara:
            tempValue=g.value.view(-1)
            tempGrad=g.grad.view(-1)
            for i in range(tempValue.shape[0]):
                    gradStore=tempValue[i].item()
                    tempValue[i]=gradStore+step
                    self.computeValue(isUpdataBnPara=False)
                    y1=self.getLossValue()
                    tempValue[i]=gradStore-step
                    self.computeValue(isUpdataBnPara=False)
                    y2=self.getLossValue()
                    tempValue[i]=gradStore
                    tempGrad[i]=(y1-y2)/2.0/step