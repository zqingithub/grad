import torchvision
from matplotlib import pyplot
import torch
from gradMatrix import gradMatrix as GM
import grad
import random
from enum import Enum
import math


class actFunc(Enum):
    sigmoid=0
    Non=1
    exp=2
    ReLU=3
    norm=4
    softMax=5
    LReLU=6

class typeOfLayer(Enum):
    input=0
    conv=1
    link=2
    pool=3

class poolMethod(Enum):
    maxPool=0
    avePool=1



def sigmoid(input):
    return input.exp()/(input.exp()+GM(torch.ones(input.value.shape,dtype=torch.float)))

def exp(input):
    return input.exp()

def ReLU(input):
    return input.ReLU()

def LReLU(input):
    return input.LReLU()

def norm(input):
    input=input.abs()
    tempSum=input@GM(torch.ones(input.value.shape,dtype=torch.float).T)
    if tempSum.value[0,0].item()==0:
        return input
    return input/tempSum.repeat(input.value.shape)

def softMax(input):
    return norm(input.exp())

def doActivation(input,actFun):
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

class dataSet:
    def __init__(self,x=None,y=None):
        self.x=x
        self.y=y

def getDataSet():
    trainDS=torchvision.datasets.MNIST(root="C:/data",train=True,download=False)
    nSample=len(trainDS.targets)
    trainDS.data=trainDS.data.cuda()
    trainDS.targets=trainDS.targets.cuda()
    td=trainDS.data/torch.ones(trainDS.data.shape,dtype=torch.float)/255
    tl=torch.zeros((nSample,10),dtype=torch.float)
    tl.scatter_(1,trainDS.targets.unsqueeze(1),torch.ones((nSample,1),dtype=torch.float))
    td=torch.flatten(td,1)
    for i in range(nSample):
        pos=random.randint(i,nSample-1)
        ex=td[pos]
        td[pos]=td[i]
        td[i]=ex
        ex=tl[pos]
        tl[pos]=tl[i]
        tl[i]=ex

    DS1=dataSet(td,tl)
    testDS=torchvision.datasets.MNIST(root="C:/data",train=False,download=False)
    nSample=len(testDS.targets)
    testDS.data=testDS.data.cuda()
    testDS.targets=testDS.targets.cuda()
    td=testDS.data/torch.ones(testDS.data.shape,dtype=torch.float)/255
    tl=torch.zeros((nSample,10),dtype=torch.float)
    tl.scatter_(1,testDS.targets.unsqueeze(1),torch.ones((nSample,1),dtype=torch.float))
    td=torch.flatten(td,1)

    return (DS1,dataSet(td,tl))

'''
    figure=pyplot.figure(figsize=(8,8))
    for i in range(100):
        pos=random.randint(0,nSample-1)
        pyplot.title(tl[pos])
        pyplot.imshow(td[pos].reshape(28,28))
        pyplot.show()
'''

def computeAcc(aNN,testDS,extractNum=100):
    nSample=testDS.x.shape[0]
    sum=0
    pos=random.randint(0,nSample-1)
    for i in range(extractNum):
        aNN.setX(testDS.x[(pos+i)%nSample].unsqueeze(0))
        aNN.computeValue()
        if torch.argmax(aNN.getY())==torch.argmax(testDS.y[(pos+i)%nSample]):
            sum=sum+1
    return sum*1.0/extractNum


def trainBySameStep(aNN,trainDS,testDS,batchSize,step,iteNum):
    nSample=trainDS.x.shape[0]
    nModel=len(aNN.paraModel)
    ite=0
    gradSum=[type(aNN.paraModel[0].grad)]*nModel
    gradPre=[type(aNN.paraModel[0].grad)]*nModel
    preIndexTrainsition=0.01
    preIndex=0.9
    riseTolerate=10
    curRiseN=0
    trainsitionRate=0.1
    preLossValue=9999999999999.0
    zeroAppro=0.00001

    for i in range(nModel):
        gradSum[i]=torch.zeros(aNN.paraModel[i].grad.shape,dtype=torch.float)
        gradPre[i]=torch.zeros(aNN.paraModel[i].grad.shape,dtype=torch.float)

    for i in range(iteNum):
        
        for k in gradSum:
            k.zero_()

        minRound=10
        lossValueSum=0.0

        for m in range(minRound):
            tempItem=ite
            lossValueSum=0.0
            for k in gradSum:
                k.zero_()
            for j in range(batchSize):

                aNN.setX(trainDS.x[tempItem%nSample].unsqueeze(0))
                aNN.setRealY(trainDS.y[tempItem%nSample].unsqueeze(0))
                aNN.computeValue()
                lossValueSum+=aNN.getLossValue()
                aNN.computeGrad()
                for k in range(nModel):
                    gradSum[k]=gradSum[k]+aNN.paraModel[k].grad
                tempItem=tempItem+1

            for k in range(nModel):
                gradPre[k]=gradPre[k]*preIndex+(1-preIndex)*step/batchSize*gradSum[k]
                aNN.paraModel[k].value=aNN.paraModel[k].value-gradPre[k]

            lossValueSum=lossValueSum/batchSize
            print("round:",i,":",m,"lossValue:",lossValueSum)
            if lossValueSum>preLossValue:
                curRiseN+=1
            else:
                curRiseN-=1
            if curRiseN<0:
                curRiseN=0
            if curRiseN>riseTolerate:
                curRiseN=0
                for g in aNN.paraModel:
                    tempIndex=torch.abs(g.grad)<zeroAppro
                    tempRate=trainsitionRate*(torch.randint(0,2,g.value.shape)*2-1)+1
                    g.value[tempIndex]=g.value[tempIndex]*tempRate[tempIndex]
                for g in range(nModel):
                    gradPre[g]=gradPre[g]*preIndexTrainsition
                print("trainsition")

            preLossValue=lossValueSum

        ite=ite+batchSize


        

        if i%100==1:
            print("acc:",computeAcc(aNN,testDS)*100,"%")
        #if i%10000==1:
        #   print("acc:",computeAcc(aNN,testDS,testDS.x.shape[0]))

        '''
        if i%1000==0:
            pyplot.figure(figsize=(8,8))
            pyplot.title(aNN.realY.value.cpu())
            pyplot.imshow(aNN.x.value.reshape(28,28).cpu())
            pyplot.show()
        '''
    print("acc:",computeAcc(aNN,testDS,testDS.x.shape[0])*100,"%")


        

#全连接神经网络 Feedforward Neural Network
class FNN:
    def __init__(self,numNodes,actFun=None):
        self.numLayer=len(numNodes)
        self.x=GM(torch.zeros((1,numNodes[0]),dtype=torch.float))
        self.y=None
        self.realY=GM(torch.zeros((1,numNodes[self.numLayer-1]),dtype=torch.float))
        self.numNodes=numNodes
        self.paraModel=[GM]*(self.numLayer-1)
        for i in range(self.numLayer-1):
            self.paraModel[i]=GM(torch.zeros((self.numNodes[i]+1,self.numNodes[i+1]),dtype=torch.float),grad.Type.gradVariable)
            self.paraModel[i].value+=torch.rand(self.paraModel[i].value.shape,dtype=torch.float)/(self.numNodes[i]+1)
            #self.paraModel[i]=GM(torch.zeros((self.numNodes[i]+1,self.numNodes[i+1]),dtype=torch.float),grad.Type.gradVariable)
        #self.paraModel[self.numLayer-2].value-=0.5
        self.actFun=actFun
        if self.actFun==None:
            self.actFun=[actFunc.sigmoid]*(self.numLayer-1)
        self.y=self.x
        for i in range(self.numLayer-1):
            self.y=GM.cat(self.y,GM(torch.randn((1,1),dtype=torch.float)),1)
            self.y=self.y@self.paraModel[i]
            self.y=doActivation(self.y,actFun[i])

        #self.y=norm(self.y)
        self.lossValue=(self.y-self.realY)@((self.y-self.realY).transpose())

    def setX(self,value):
        self.x.value=value

    def setRealY(self,value):
        self.realY.value=value

    def computeValue(self):
        self.lossValue.computeValue()

    def computeGrad(self):
        self.lossValue.computeGrad(self.paraModel)

    def getLossValue(self):
        return self.lossValue.value[0,0].item()

    def getY(self):
        return self.y.value


class FNN_res:
    def __init__(self,numNodes,actFun=None):
        self.numLayer=len(numNodes)
        self.x=GM(torch.zeros((1,numNodes[0]),dtype=torch.float))
        self.y=None
        tempY=GM(torch.zeros((1,1),dtype=torch.float))
        numBeforeOutput=1+sum(numNodes)-numNodes[0]-numNodes[-1]
        self.realY=GM(torch.zeros((1,numNodes[self.numLayer-1]),dtype=torch.float))
        self.numNodes=numNodes
        self.paraModel=[GM]*(self.numLayer-1)
        for i in range(self.numLayer-1):
            self.paraModel[i]=GM(torch.zeros((self.numNodes[i]+1,self.numNodes[i+1]),dtype=torch.float),grad.Type.gradVariable)
            self.paraModel[i].value+=torch.rand(self.paraModel[i].value.shape,dtype=torch.float)/(self.numNodes[i]+1)
            #self.paraModel[i]=GM(torch.zeros((self.numNodes[i]+1,self.numNodes[i+1]),dtype=torch.float),grad.Type.gradVariable)
        #self.paraModel[self.numLayer-2].value-=0.5
        if self.numLayer>3:
            self.paraModel[-1]=GM(torch.zeros((numBeforeOutput+1,self.numNodes[-1]),dtype=torch.float),grad.Type.gradVariable)
            self.paraModel[-1].value+=torch.rand(self.paraModel[-1].value.shape,dtype=torch.float)/(numBeforeOutput+1)
        self.actFun=actFun
        if self.actFun==None:
            self.actFun=[actFunc.sigmoid]*(self.numLayer-1)
        self.y=self.x
        for i in range(self.numLayer-1):
            self.y=GM.cat(self.y,GM(torch.randn((1,1),dtype=torch.float)),1)
            self.y=self.y@self.paraModel[i]
            self.y=doActivation(self.y,actFun[i])
            if i<=self.numLayer-3:
                tempY=GM.cat(tempY,self.y,1)
            if i==self.numLayer-3 and self.numLayer>3:
                self.y=tempY

        #self.y=norm(self.y)
        self.lossValue=(self.y-self.realY)@((self.y-self.realY).transpose())

    def setX(self,value):
        self.x.value=value

    def setRealY(self,value):
        self.realY.value=value

    def computeValue(self):
        self.lossValue.computeValue()

    def computeGrad(self):
        self.lossValue.computeGrad(self.paraModel)

    def getLossValue(self):
        return self.lossValue.value[0,0].item()

    def getY(self):
        return self.y.value

class layer:
    def __init__(self,layer=typeOfLayer.conv,outputDim=0,kernelSize=0,\
        channelSize=0,actFunc=actFunc.LReLU,poolMethod=poolMethod.avePool,channelTable=None,isShortCircuit=False,isRecurrent=False):
        self.typeOfLayer=layer
        self.outputDim=outputDim
        self.kernelSize=kernelSize
        self.channelSize=channelSize
        self.actFunc=actFunc
        self.poolMethod=poolMethod
        self.channelTable=channelTable
        self.isShortCircuit=isShortCircuit
        self.isRecurrent=isRecurrent

#卷积神经网络 Convolutional Neural Network
class CNN:
    def __init__(self,layer):
        self.x=None
        self.realY=None
        self.y=None
        self.paraModel=[]
        self.lossValue=None
        numLayer=len(layer)
        for iterLayer in range(numLayer):
            if layer[iterLayer].typeOfLayer==typeOfLayer.input:
                self.x=GM(torch.zeros((1,layer[iterLayer].outputDim),dtype=torch.float))
                outputSize=math.isqrt(layer[iterLayer].outputDim)
                layer[iterLayer].channelSize=1
                self.y=self.x.reshape([1,1,outputSize,outputSize])
            if layer[iterLayer].typeOfLayer==typeOfLayer.conv:
                if not(layer[iterLayer-1].typeOfLayer==typeOfLayer.conv or \
                       layer[iterLayer-1].typeOfLayer==typeOfLayer.pool or \
                       layer[iterLayer-1].typeOfLayer==typeOfLayer.input):
                    raise TypeError("conv pre-layer is not right")
                mask=torch.ones(layer[iterLayer].channelSize,layer[iterLayer-1].channelSize,\
                                layer[iterLayer].kernelSize,layer[iterLayer].kernelSize,dtype=torch.float)
                self.paraModel.append(GM(torch.rand(mask.shape,dtype=torch.float),grad.Type.gradVariable))
                if layer[iterLayer].channelTable!=None:
                    for i in range(layer[iterLayer].channelSize):
                        for j in range(layer[iterLayer-1].channelSize):
                            if layer[iterLayer].channelTable[i][j]<1:
                                mask[i,j].zero_()
                for i in range(layer[iterLayer].channelSize):
                    self.paraModel[-1].value[i]=self.paraModel[-1].value[i]/(torch.sum(mask[i])+1)
                kernel=self.paraModel[-1]
                if layer[iterLayer].channelTable!=None:
                    kernel=kernel*GM(mask)
                self.y=self.y.conv2D(kernel)
                self.paraModel.append(GM(torch.rand(1,layer[iterLayer].channelSize,1,1,\
                    dtype=torch.float)/(torch.sum(mask[i])+1),grad.Type.gradVariable))
                self.y=self.y+self.paraModel[-1].repeat([1,1,self.y.value.shape[2],self.y.value.shape[3]])
                self.y=doActivation(self.y,layer[iterLayer].actFunc)

            if layer[iterLayer].typeOfLayer==typeOfLayer.pool:
                if not(layer[iterLayer-1].typeOfLayer==typeOfLayer.conv):
                    raise TypeError("pool pre-layer is not right")
                layer[iterLayer].channelSize=layer[iterLayer-1].channelSize
                self.y=self.__doPool(self.y,layer[iterLayer].poolMethod,layer[iterLayer].kernelSize)
                if layer[iterLayer].actFunc!=actFunc.Non:
                    self.paraModel.append(GM(torch.rand(1,layer[iterLayer].channelSize,\
                        1,1,dtype=torch.float)/2,grad.Type.gradVariable))
                    self.y=self.y*(self.paraModel[-1].repeat([1,1,self.y.value.shape[2],self.y.value.shape[3]]))
                    self.paraModel.append(GM(torch.rand(1,layer[iterLayer].channelSize,\
                        1,1,dtype=torch.float)/2,grad.Type.gradVariable))
                    self.y=self.y+(self.paraModel[-1].repeat([1,1,self.y.value.shape[2],self.y.value.shape[3]]))
                    self.y=doActivation(self.y,layer[iterLayer].actFunc)

            if layer[iterLayer].typeOfLayer==typeOfLayer.link:
                if layer[iterLayer-1].typeOfLayer==typeOfLayer.input:
                    raise TypeError("link pre-layer is not right")
                if layer[iterLayer-1].typeOfLayer==typeOfLayer.conv or layer[iterLayer-1].typeOfLayer==typeOfLayer.pool:
                    self.y=self.y.reshape([1,self.y.value.shape[1]*self.y.value.shape[2]*self.y.value.shape[3]])
                self.y=GM.cat(self.y,GM(torch.ones(1,1,dtype=torch.float)),1)
                self.paraModel.append(GM(torch.rand(self.y.value.shape[1],\
                                layer[iterLayer].outputDim,dtype=torch.float)/self.y.value.shape[1],grad.Type.gradVariable))
                self.y=doActivation(self.y@self.paraModel[-1],layer[iterLayer].actFunc)

        self.realY=GM(torch.zeros(self.y.value.shape,dtype=torch.float))
        self.lossValue=(self.y-self.realY)@((self.y-self.realY).transpose())

    def __doPool(self,y,method,kernelSize):
        if method==poolMethod.avePool:
            return y.avgPool2D(kernelSize)
        if method==poolMethod.maxPool:
            return y.maxPool2D(kernelSize)

    def setX(self,value):
        self.x.value=value

    def setRealY(self,value):
        self.realY.value=value

    def computeValue(self):
        self.lossValue.computeValue()

    def computeGrad(self):
        self.lossValue.computeGrad(self.paraModel)

    def getLossValue(self):
        return self.lossValue.value[0,0].item()

    def getY(self):
        return self.y.value

#循环神经网络 Recurrent Neural Network
'''
class RNN:
    def __init__(self,layer,numOfMaxRecurrent=10):
        layer[-1].isRecurrent=False
        self.nowRecurrent=0
        self.x=[GM()]*numOfMaxRecurrent
        self.realY=[GM()]*numOfMaxRecurrent
        self.y=[GM()]*numOfMaxRecurrent
        self.recurrentCat=[]
        self.recurrentOutput=[]
        for i in range(numOfMaxRecurrent):
            self.recurrentCat.append([])
            self.recurrentOutput.append([])
        self.paraModel=[]
        self.lossValue=[GM()]*numOfMaxRecurrent
        numLayer=len(layer)
        shortCircuitVariable=[]
        for iterLayer in range(numLayer):
            if layer[iterLayer].typeOfLayer==typeOfLayer.input:
                self.x[0]=GM(torch.zeros((1,layer[iterLayer].outputDim),dtype=torch.float))
                self.y[0]=self.x[0]

            if layer[iterLayer].typeOfLayer==typeOfLayer.link:
                self.y[0]=GM.cat(self.y[0],GM(torch.ones(1,1,dtype=torch.float)),1)
                if layer[iterLayer].isRecurrent==True:
                    self.y[0]=GM.cat(GM(torch.zeros(1,layer[iterLayer].outputDim,dtype=torch.float)),self.y[0],1)
                    self.recurrentCat[0].append(self.y[0])
                self.paraModel.append(GM(torch.rand(self.y[0].value.shape[1],\
                                layer[iterLayer].outputDim,dtype=torch.float)/self.y[0].value.shape[1],grad.Type.gradVariable))
                self.y[0]=doActivation(self.y[0]@self.paraModel[-1],layer[iterLayer].actFunc)
                if layer[iterLayer].isRecurrent==True:
                    self.recurrentOutput[0].append(self.y[0])
                if iterLayer==numLayer-2:
                    for sv in shortCircuitVariable:
                        self.y[0]=GM.cat(self.y[0],sv,1)
                if layer[iterLayer].isShortCircuit==True:
                    shortCircuitVariable.append(self.y[0])
        self.realY[0]=GM(torch.zeros(self.y[0].value.shape,dtype=torch.float))
        self.lossValue[0]=(self.y[0]-self.realY[0])@((self.y[0]-self.realY[0]).transpose())

        twinsDict={self.x[0]:self.x[0],self.y[0]:self.y[0],self.realY[0]:self.realY[0],self.lossValue[0]:self.lossValue[0]}
        for re in self.recurrentCat[0]:
            twinsDict[re]=re
        for re in self.recurrentOutput[0]:
            twinsDict[re]=re

        for iterRecurrent in range(1,numOfMaxRecurrent):
            GM.copy(graph=self.lossValue[0],twinsDict=twinsDict)
            self.x[iterRecurrent]=twinsDict[self.x[0]]
            self.y[iterRecurrent]=twinsDict[self.y[0]]
            self.realY[iterRecurrent]=twinsDict[self.realY[0]]
            self.lossValue[iterRecurrent]=twinsDict[self.lossValue[0]]
            for re in self.recurrentCat[0]:
                self.recurrentCat[iterRecurrent].append(twinsDict[re])
            for re in self.recurrentOutput[0]:
                self.recurrentOutput[iterRecurrent].append(twinsDict[re])

        for iterRecurrent in range(1,numOfMaxRecurrent):
            for c in range(len(self.recurrentCat[0])):
                self.recurrentCat[iterRecurrent][c].left=self.recurrentOutput[iterRecurrent-1][c]
                self.recurrentCat[iterRecurrent][c].left.numOfShare+=1
            self.lossValue[iterRecurrent]=(self.lossValue[iterRecurrent-1]*(GM(torch.ones(1,1,dtype=torch.float)*iterRecurrent))\
                +self.lossValue[iterRecurrent])/(GM(torch.ones(1,1,dtype=torch.float)*(iterRecurrent+1)))


    def __computeValue(self,recurrentIndex):
        if recurrentIndex==len(self.x)-1:
            self.lossValue[recurrentIndex].computeValue()
            return 
        for re in self.recurrentOutput[recurrentIndex]:
            re.numOfShare-=1
        self.lossValue[recurrentIndex].computeValue()
        for re in self.recurrentOutput[recurrentIndex]:
            re.numOfShare+=1

    def __computeGrad(self,recurrentIndex):
        if recurrentIndex==len(self.x)-1:
            self.lossValue[recurrentIndex].computeGrad(self.paraModel)
            return 

        for re in self.recurrentOutput[recurrentIndex]:
            re.numOfShare-=1
        self.lossValue[recurrentIndex].computeGrad(self.paraModel)
        for re in self.recurrentOutput[recurrentIndex]:
            re.numOfShare+=1

    def setX(self,value):
        tempX=value.reshape([28,28])
        self.nowRecurrent=4
        for i in range(self.nowRecurrent):
            self.x[i].value=tempX[:,i:i+7].reshape(1,196)

    def setRealY(self,value):
        for i in range(0,self.nowRecurrent):
            self.realY[i].value=value

    def computeValue(self):
        self.__computeValue(self.nowRecurrent-1)

    def computeGrad(self):
        self.__computeGrad(self.nowRecurrent-1)

    def getLossValue(self):
        return self.lossValue[self.nowRecurrent-1].value[0,0].item()

    def getY(self):
        sum=torch.zeros(self.y[0].value.shape,dtype=torch.float)
        for i in range(0,self.nowRecurrent):
            sum+=self.y[i].value
        return sum/4
'''
class RNN:
    def __init__(self,layer,numOfMaxRecurrent=10):
        #layer[-1].isRecurrent=False
        numOfMaxRecurrent+=1
        self.nowRecurrent=0
        self.x=None
        self.realY=None
        self.y=None
        self.xList=[]
        self.realYList=[]
        self.yList=[]
        self.recurrentForwordInput=[]
        self.forwordStore=[]
        self.recurrentOutput=[]
        self.lossValue=None
        self.recurrentZero=[]
        self.paraModel=[]
        self.paraModelStore=[]
        self.allPara=[]
        self.preGrad=[]
        self.lossValueSum=GM(torch.zeros(1,1,dtype=torch.float))
  
        numLayer=len(layer)
        shortCircuitVariable=[]
        for iterLayer in range(numLayer):
            if layer[iterLayer].typeOfLayer==typeOfLayer.input:
                self.x=GM(torch.zeros((1,layer[iterLayer].outputDim),dtype=torch.float))
                self.y=self.x

            if layer[iterLayer].typeOfLayer==typeOfLayer.link:
                if layer[iterLayer].isRecurrent==True:
                    self.recurrentForwordInput.append(GM(torch.zeros(1,layer[iterLayer].outputDim,dtype=torch.float),type=grad.Type.gradVariable))
                    self.recurrentZero.append(GM(torch.zeros(1,layer[iterLayer].outputDim,dtype=torch.float)))
                    self.y=GM.cat(self.recurrentForwordInput[-1],self.y,1)
                self.y=GM.cat(self.y,GM(torch.ones(1,1,dtype=torch.float)),1)
                self.paraModel.append(GM(torch.rand(self.y.value.shape[1],\
                                layer[iterLayer].outputDim,dtype=torch.float)/self.y.value.shape[1],grad.Type.gradVariable))
                self.y=doActivation(self.y@self.paraModel[-1],layer[iterLayer].actFunc)
                if layer[iterLayer].isRecurrent==True:
                    self.recurrentOutput.append(self.y)
                if iterLayer==numLayer-2:
                    for sv in shortCircuitVariable:
                        self.y=GM.cat(self.y,sv,1)
                if layer[iterLayer].isShortCircuit==True:
                    shortCircuitVariable.append(self.y)
        self.realY=GM(torch.zeros(self.y.value.shape,dtype=torch.float))
        self.lossValue=(self.y-self.realY)@((self.y-self.realY).transpose())


        for ite in self.paraModel:
            self.paraModelStore.append(GM(torch.zeros(ite.value.shape,dtype=torch.float),type=grad.Type.gradVariable))

        for ite in self.recurrentZero:
            self.preGrad.append(GM(torch.zeros(ite.value.shape,dtype=torch.float),type=grad.Type.gradVariable))

        for ite in self.paraModel:
            self.allPara.append(ite)
       
        for ite in self.recurrentForwordInput:
            self.allPara.append(ite)

        for i in range(numOfMaxRecurrent):
            self.forwordStore.append([])
            self.xList.append(GM(torch.zeros(self.x.value.shape,dtype=torch.float)))
            self.yList.append(GM(torch.zeros(self.y.value.shape,dtype=torch.float)))
            self.realYList.append(GM(torch.zeros(self.realY.value.shape,dtype=torch.float)))
            for j in range(len(self.recurrentZero)):
                self.forwordStore[i].append(GM(torch.zeros(self.recurrentZero[j].value.shape,dtype=torch.float)))  

    def __loadValue(self,target,value):
        for i in range(len(self.recurrentZero)):
            target[i].value=value[i].value.clone()

    def __loadXRealY(self,index):
        self.x.value=self.xList[index].value.clone()
        self.realY.value=self.realYList[index].value.clone()

    def __recordY(self,index):
        self.yList[index].value=self.y.value.clone()

    def __loadPreGrad(self):
        for i in range(len(self.recurrentZero)):
            self.recurrentOutput[i].sumOfGrad+=self.preGrad[i].grad

    def __updatePreGrad(self):
        for i in range(len(self.recurrentZero)):
            self.preGrad[i].grad+=self.recurrentForwordInput[i].grad

    def __zeroPreGrad(self):
        for iter in self.preGrad:
            iter.grad.zero_()

    def computeValue(self):

        self.__loadValue(self.forwordStore[0],self.recurrentZero)
        self.lossValueSum.value.zero_()

        for i in range(self.nowRecurrent):
            self.__loadValue(self.recurrentForwordInput,self.forwordStore[i])
            self.__loadXRealY(i)
            self.lossValue.computeValue()
            self.lossValueSum.value+=self.lossValue.value
            self.__recordY(i)
            self.__loadValue(self.forwordStore[i+1],self.recurrentOutput)


    def __updateParaModelStore(self):
        for i in range(len(self.paraModel)):
            self.paraModelStore[i].grad+=self.paraModel[i].grad

    def __zeroParaModelStore(self):
        for ite in self.paraModelStore:
            ite.grad.zero_()
        
         
    def __computeGradHaveRecurrent(self):

        self.__zeroPreGrad()
        self.__zeroParaModelStore()

        for i in range(self.nowRecurrent-1,-1,-1):
            self.__loadValue(self.recurrentForwordInput,self.forwordStore[i])
            self.__loadXRealY(i)
            self.lossValue.computeValue()
            self.__loadPreGrad()
            self.lossValue.computeGrad(self.allPara)
            self.__updateParaModelStore()
            self.__zeroPreGrad()
            self.__updatePreGrad()

        for i in range(len(self.paraModel)):
            self.paraModel[i].grad=self.paraModelStore[i].grad*1.0/self.nowRecurrent

    def computeGrad(self):
        if len(self.recurrentZero)>0:
            self.__computeGradHaveRecurrent()
        else:
            self.__zeroParaModelStore()
            for i in range(self.nowRecurrent):
                self.__loadXRealY(i)
                self.lossValue.computeValue()
                self.lossValue.computeGrad(self.paraModel)
                self.__updateParaModelStore()
            for i in range(len(self.paraModel)):
                self.paraModel[i].grad=self.paraModelStore[i].grad*1.0/self.nowRecurrent
                

    def setX(self,value):
        tempX=value.reshape([28,28])
        self.nowRecurrent=4
        for i in range(self.nowRecurrent):
            self.xList[i].value=tempX[:,i:i+7].reshape(1,196)

    def setRealY(self,value):
        for i in range(0,self.nowRecurrent):
            self.realYList[i].value=value

    def getLossValue(self):
        return self.lossValueSum.value[0,0].item()*1.0/self.nowRecurrent

    def getY(self):
        sum=torch.zeros(self.y.value.shape,dtype=torch.float)
        for i in range(0,self.nowRecurrent):
            sum+=self.yList[i].value
        return sum/self.nowRecurrent

    def computeGradByEstimate(self,gradVariable,step):
        for g in gradVariable:
            tempValue=g.value.view(-1)
            tempGrad=g.grad.view(-1)
            for i in range(tempValue.shape[0]):
                    gradStore=tempValue[i].item()
                    tempValue[i]=gradStore+step
                    self.computeValue()
                    y1=self.getLossValue()
                    tempValue[i]=gradStore-step
                    self.computeValue()
                    y2=self.getLossValue()
                    tempValue[i]=gradStore
                    tempGrad[i]=(y1-y2)/2.0/step


#循环神经网络 Bidirectional Recurrent Neural Network
class RNN_Bi:
    def __init__(self,layer,numOfMaxRecurrent=10):
        #layer[-1].isRecurrent=False
        numOfMaxRecurrent+=1
        self.nowRecurrent=0
        self.x=None
        self.realY=None
        self.y=None
        self.xList=[]
        self.realYList=[]
        self.yList=[]
        self.recurrentForwordInput=[]
        self.recurrentBackwordInput=[]
        self.forwordStore=[]
        self.backwordStore=[]
        self.recurrentOutput=[]
        self.lossValue=None
        self.recurrentZero=[]
        self.paraModel=[]
        self.paraModelStore=[]
        self.allPara=[]
        self.preGrad=[]
        self.lossValueSum=GM(torch.zeros(1,1,dtype=torch.float))
  
        numLayer=len(layer)
        shortCircuitVariable=[]
        for iterLayer in range(numLayer):
            if layer[iterLayer].typeOfLayer==typeOfLayer.input:
                self.x=GM(torch.zeros((1,layer[iterLayer].outputDim),dtype=torch.float))
                self.y=self.x

            if layer[iterLayer].typeOfLayer==typeOfLayer.link:
                if layer[iterLayer].isRecurrent==True:
                    self.recurrentForwordInput.append(GM(torch.zeros(1,layer[iterLayer].outputDim,dtype=torch.float),type=grad.Type.gradVariable))
                    self.recurrentZero.append(GM(torch.zeros(1,layer[iterLayer].outputDim,dtype=torch.float)))
                    self.y=GM.cat(self.recurrentForwordInput[-1],self.y,1)
                    self.recurrentBackwordInput.append(GM(torch.zeros(1,layer[iterLayer].outputDim,dtype=torch.float),type=grad.Type.gradVariable))
                    self.y=GM.cat(self.y,self.recurrentBackwordInput[-1],1)
                self.y=GM.cat(self.y,GM(torch.ones(1,1,dtype=torch.float)),1)
                self.paraModel.append(GM(torch.rand(self.y.value.shape[1],\
                                layer[iterLayer].outputDim,dtype=torch.float)/self.y.value.shape[1],grad.Type.gradVariable))
                self.y=doActivation(self.y@self.paraModel[-1],layer[iterLayer].actFunc)
                if layer[iterLayer].isRecurrent==True:
                    self.recurrentOutput.append(self.y)
                if iterLayer==numLayer-2:
                    for sv in shortCircuitVariable:
                        self.y=GM.cat(self.y,sv,1)
                if layer[iterLayer].isShortCircuit==True:
                    shortCircuitVariable.append(self.y)
        self.realY=GM(torch.zeros(self.y.value.shape,dtype=torch.float))
        self.lossValue=(self.y-self.realY)@((self.y-self.realY).transpose())


        for ite in self.paraModel:
            self.paraModelStore.append(GM(torch.zeros(ite.value.shape,dtype=torch.float),type=grad.Type.gradVariable))

        for ite in self.recurrentZero:
            self.preGrad.append(GM(torch.zeros(ite.value.shape,dtype=torch.float),type=grad.Type.gradVariable))

        for ite in self.paraModel:
            self.allPara.append(ite)
       
        for ite in self.recurrentForwordInput:
            self.allPara.append(ite)

        for ite in self.recurrentBackwordInput:
            self.allPara.append(ite)

        for i in range(numOfMaxRecurrent):
            self.forwordStore.append([])
            self.backwordStore.append([])
            self.xList.append(GM(torch.zeros(self.x.value.shape,dtype=torch.float)))
            self.yList.append(GM(torch.zeros(self.y.value.shape,dtype=torch.float)))
            self.realYList.append(GM(torch.zeros(self.realY.value.shape,dtype=torch.float)))
            for j in range(len(self.recurrentZero)):
                self.forwordStore[i].append(GM(torch.zeros(self.recurrentZero[j].value.shape,dtype=torch.float)))
                self.backwordStore[i].append(GM(torch.zeros(self.recurrentZero[j].value.shape,dtype=torch.float)))     

    def __loadValue(self,target,value):
        for i in range(len(self.recurrentZero)):
            target[i].value=value[i].value.clone()

    def __loadXRealY(self,index):
        self.x.value=self.xList[index].value.clone()
        self.realY.value=self.realYList[index].value.clone()

    def __recordY(self,index):
        self.yList[index].value=self.y.value.clone()

    def __loadPreGrad(self):
        for i in range(len(self.recurrentZero)):
            self.recurrentOutput[i].sumOfGrad+=self.preGrad[i].grad

    def __updatePreGrad(self,isForword=True):
        if isForword:
            for i in range(len(self.recurrentZero)):
                self.preGrad[i].grad+=self.recurrentForwordInput[i].grad
        else:
            for i in range(len(self.recurrentZero)):
                self.preGrad[i].grad+=self.recurrentBackwordInput[i].grad

    def __zeroPreGrad(self):
        for iter in self.preGrad:
            iter.grad.zero_()

    def computeValue(self):
        self.__loadValue(self.forwordStore[0],self.recurrentZero)
        self.__loadValue(self.recurrentBackwordInput,self.recurrentZero)
        self.lossValueSum.value.zero_()
        for i in range(self.nowRecurrent):
            self.__loadValue(self.recurrentForwordInput,self.forwordStore[i])
            self.__loadXRealY(i)
            self.lossValue.computeValue()
            self.__loadValue(self.forwordStore[i+1],self.recurrentOutput)

        self.__loadValue(self.backwordStore[self.nowRecurrent-1],self.recurrentZero)
        self.__loadValue(self.recurrentForwordInput,self.recurrentZero)
        for i in range(self.nowRecurrent-1,-1,-1):
            self.__loadValue(self.recurrentBackwordInput,self.backwordStore[i])
            self.__loadXRealY(i)
            self.lossValue.computeValue()
            self.__loadValue(self.backwordStore[i-1],self.recurrentOutput)

        for i in range(self.nowRecurrent):
            self.__loadValue(self.recurrentForwordInput,self.forwordStore[i])
            self.__loadValue(self.recurrentBackwordInput,self.backwordStore[i])
            self.__loadXRealY(i)
            self.lossValue.computeValue()
            self.__recordY(i)
            self.lossValueSum.value+=self.lossValue.value

    def __updateParaModelStore(self):
        for i in range(len(self.paraModel)):
            self.paraModelStore[i].grad+=self.paraModel[i].grad

    def __zeroParaModelStore(self):
        for ite in self.paraModelStore:
            ite.grad.zero_()
        
          

    def __computeGradHaveRecurrent(self):
        self.__zeroPreGrad()
        self.__zeroParaModelStore()

        for i in range(self.nowRecurrent):
            self.__loadValue(self.recurrentForwordInput,self.recurrentZero)
            self.__loadValue(self.recurrentBackwordInput,self.backwordStore[i])
            self.__loadXRealY(i)
            self.lossValue.computeValue()
            self.__loadPreGrad()
            self.lossValue.computeGrad(self.allPara,torch.zeros(self.lossValue.value.shape,dtype=torch.float))
            #self.recurrentOutput[-1].computeGrad(self.allPara,torch.zeros(self.recurrentOutput[-1].value.shape,dtype=torch.float))
            self.__updateParaModelStore()
            self.__zeroPreGrad()
            self.__updatePreGrad(isForword=False)
            self.__loadValue(self.recurrentForwordInput,self.forwordStore[i])
            self.lossValue.computeValue()
            self.lossValue.computeGrad(self.allPara)
            self.__updateParaModelStore()
            self.__updatePreGrad(isForword=False)


        self.__zeroPreGrad()
        for i in range(self.nowRecurrent-1,-1,-1):
            self.__loadValue(self.recurrentBackwordInput,self.recurrentZero)
            self.__loadValue(self.recurrentForwordInput,self.forwordStore[i])
            self.__loadXRealY(i)
            self.lossValue.computeValue()
            self.__loadPreGrad()
            self.lossValue.computeGrad(self.allPara,torch.zeros(self.lossValue.value.shape,dtype=torch.float))
            #self.recurrentOutput[-1].computeGrad(self.allPara,torch.zeros(self.recurrentOutput[-1].value.shape,dtype=torch.float))
            self.__updateParaModelStore()
            self.__zeroPreGrad()
            self.__updatePreGrad(isForword=True)
            self.__loadValue(self.recurrentBackwordInput,self.backwordStore[i])
            self.lossValue.computeValue()
            self.lossValue.computeGrad(self.allPara)
            self.__updatePreGrad(isForword=True)

        for i in range(len(self.paraModel)):
            self.paraModel[i].grad=self.paraModelStore[i].grad*1.0/self.nowRecurrent

    def computeGrad(self):
        if len(self.recurrentZero)>0:
            self.__computeGradHaveRecurrent()
        else:
            self.__zeroParaModelStore()
            for i in range(self.nowRecurrent):
                self.__loadXRealY(i)
                self.lossValue.computeValue()
                self.lossValue.computeGrad(self.paraModel)
                self.__updateParaModelStore()
            for i in range(len(self.paraModel)):
                self.paraModel[i].grad=self.paraModelStore[i].grad*1.0/self.nowRecurrent
                

    def setX(self,value):
        tempX=value.reshape([28,28])
        self.nowRecurrent=4
        for i in range(self.nowRecurrent):
            self.xList[i].value=tempX[:,i:i+7].reshape(1,196)

    def setRealY(self,value):
        for i in range(0,self.nowRecurrent):
            self.realYList[i].value=value

    def getLossValue(self):
        return self.lossValueSum.value[0,0].item()*1.0/self.nowRecurrent

    def getY(self):
        sum=torch.zeros(self.y.value.shape,dtype=torch.float)
        for i in range(0,self.nowRecurrent):
            sum+=self.yList[i].value
        return sum/self.nowRecurrent

    def computeGradByEstimate(self,gradVariable,step):
        for g in gradVariable:
            tempValue=g.value.view(-1)
            tempGrad=g.grad.view(-1)
            for i in range(tempValue.shape[0]):
                    gradStore=tempValue[i].item()
                    tempValue[i]=gradStore+step
                    self.computeValue()
                    y1=self.getLossValue()
                    tempValue[i]=gradStore-step
                    self.computeValue()
                    y2=self.getLossValue()
                    tempValue[i]=gradStore
                    tempGrad[i]=(y1-y2)/2.0/step


