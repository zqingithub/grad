#用于实现一些辅助功能，例如数据集导入，准确率计算，训练过程可视化等
from __future__ import annotations
from ast import Dict, List, Tuple
from multiprocessing import Queue
import time
from typing import Any
import numpy
import torch.utils.data.dataset
import torchvision
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import random
import threading
from artificial_neural_network import ANN, actFunc, layerInfo
import math
import json
from gradMatrix import gradMatrix as GM
import artificial_neural_network
import os
import datetime

class dataSet:
    """
    用于训练时储存数据集，x为样本的特征，y为样本的标签。x的shape形状应该为：[n,t,1,c,h,w].其中：
    n：表示样本个数。
    t：每个样本的时序个数。对于循环神经网络，每个样本的特征是一个时间序列，对于非循环神经网络，则可认为只有一个时序的特征。
    1：为了兼容pytorch中卷积操作对图片格式的要求，这里固定设置为1。在pytorch中此处表示图片的个数，但是在实际实现过程中每次只处理一幅图片，因此这里固定为1。
    c：图片的通道数。
    h：图片的高。
    w：图片的宽。
    假设有100个样本，每个样本的特征是2个时间序列，每个时间序列是一幅3通道的5×7大小的图片，则x的shape为[100,2,1,3,5,7]
    y的shape形状应该为：[n,t,1,d],其中：
    n：表示样本个数。
    t：每个样本的时序个数。样本有几个时序的特征，就应该有相同时序数量的标签。
    1：固定为1。
    d：标签的维度。假设是分类问题，标签的维度就是类别的个数。假设是回归问题，标签的维度就是需要回归的目标向量的维度。
    """
    def __init__(self,x:torch.tensor=None,y:torch.tensor=None):
        self.x:torch.tensor=x
        self.y:torch.tensor=y

def getDataSet()->Tuple[dataSet]:
    trainDS:torch.utils.data.dataset=torchvision.datasets.MNIST(root="C:/data",train=True,download=False)
    nSample:int=len(trainDS.targets)
    trainDS.data=trainDS.data.cuda()
    trainDS.targets=trainDS.targets.cuda()
    td:torch.tensor=trainDS.data/torch.ones(trainDS.data.shape,dtype=torch.float)/255
    td=td.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    tl:torch.tensor=torch.zeros((nSample,10),dtype=torch.float)
    tl.scatter_(1,trainDS.targets.unsqueeze(1),torch.ones((nSample,1),dtype=torch.float))
    tl=tl.unsqueeze(1).unsqueeze(1)
    newOrder=list(range(nSample))
    for i in range(nSample):
        pos=random.randint(i,nSample-1)
        ex=newOrder[pos]
        newOrder[pos]=newOrder[i]
        newOrder[i]=ex
    td=td[newOrder]
    tl=tl[newOrder]

    DS1:dataSet=dataSet(td,tl)
    testDS:torch.utils.data.dataset=torchvision.datasets.MNIST(root="C:/data",train=False,download=False)
    nSample=len(testDS.targets)
    testDS.data=testDS.data.cuda()
    testDS.targets=testDS.targets.cuda()
    td=testDS.data/torch.ones(testDS.data.shape,dtype=torch.float)/255
    td=td.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    tl=torch.zeros((nSample,10),dtype=torch.float)
    tl.scatter_(1,testDS.targets.unsqueeze(1),torch.ones((nSample,1),dtype=torch.float))
    tl=tl.unsqueeze(1).unsqueeze(1)
    newOrder=list(range(nSample))
    for i in range(nSample):
        pos=random.randint(i,nSample-1)
        ex=newOrder[pos]
        newOrder[pos]=newOrder[i]
        newOrder[i]=ex
    td=td[newOrder]
    tl=tl[newOrder]


    return (DS1,dataSet(td,tl))

'''
    figure=pyplot.figure(figsize=(8,8))
    for i in range(100):
        pos=random.randint(0,nSample-1)
        pyplot.title(tl[pos])
        pyplot.imshow(td[pos].reshape(28,28))
        pyplot.show()
'''

def computeAcc(aNN:ANN,testDS:dataSet,extractNum:int=100,isUseFinishModel:bool=False,isUseBnParaPerTime:bool=True)->float:
    nSample:int=testDS.x.shape[0]
    sum:float=0.0
    pos:int=random.randint(0,nSample-1)
    if not isUseFinishModel:
        ite=math.ceil(extractNum/aNN.batchSize)
        for i in range(ite):
            aNN.setXBatch(testDS.x,pos)
            aNN.computeValue(isUpdataBnPara=False)
            aNN.computeYClassifier()
            for j in range(aNN.batchSize):
                if torch.argmax(aNN.yClassifier[j])==torch.argmax(aNN.getYClassifier(testDS.y[(pos+j)%nSample])):
                    sum=sum+1
            pos+=aNN.batchSize
        return sum*1.0/ite/aNN.batchSize
    else:
        for i in range(extractNum):
            tp=(i+pos)%nSample
            aNN.finishModel.setXBatch(testDS.x,tp)
            aNN.finishModel.computeValue(isUpdataBnPara=False,isUseBnParePerTime=isUseBnParaPerTime)
            aNN.finishModel.computeYClassifier()
            if torch.argmax(aNN.finishModel.yClassifier[0])==torch.argmax(aNN.finishModel.getYClassifier(testDS.y[tp])):
                sum=sum+1
        return sum*1.0/extractNum


def showTrainProcess(size:int,queue:Queue,modelName:str=""):

    x=range(size)
    y_test=[0.0]*size
    y_train=[0.0]*size
    pos=[1]

    drawLineThread:threading.Thread=threading.Thread(target=drawLine,name='drawLineThread',kwargs={'x':x,'y_test':y_test,'y_train':y_train,'pos':pos,'modelName':modelName},daemon=True)
    drawLineThread.start()

    while True:
        try:
            y_test[pos[0]]=queue.get_nowait()
            y_train[pos[0]]=queue.get_nowait()
            if y_train[pos[0]]<0:
                break
            pos[0]=(pos[0]+1)%size
        except:
            time.sleep(1)

    drawLineThread.terminate()
    

def drawLine(x:List[int],y_test:List[float],y_train:List[float],pos:List[int],modelName:str=""):

    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False   

    fig,ax=plt.subplots(figsize=(10,6))
    ax.set_xlim(0,len(x))  
    ax.set_ylim(0,1.1)
    ax.set_yticks(numpy.arange(0,1.1,0.01))
    yLabel=[""]*110
    for i in range(10,110,10):
        yLabel[i]=str(i)+"%"
    ax.set_yticklabels(yLabel)
    ax.set_xlabel('time')
    ax.set_ylabel('acc:')
    ax.set_title(modelName+' acc_record')
    ax.grid(True) 

    lineRecordTest,=ax.plot([],[],'g-',linewidth=1)
    lineRecordTrain,=ax.plot([],[],'r-',linewidth=1)
    lineUp,=ax.plot([0,len(x)],[1,1],'y-',linewidth=2)
    pointRecordTest,=ax.plot(0,0,'ro',markersize=8)
    pointRecordTrain,=ax.plot(0,0,'ro',markersize=8)
    pointLatestTest,=ax.plot(0,0,'ro',markersize=8)
    pointLatestTrain,=ax.plot(0,0,'ro',markersize=8)

    fig.legend(loc='upper right',handles=[lineRecordTest,lineRecordTrain], labels=['acc of test data set','acc of train data set'])

    pointMarker=['s','x','o','*']
    pointSize=[4,8,6,10]
    pointColor=['r','g','b','y']

    def update(time):

        showPos=(pos[0]-1)%len(x)
        lineRecordTest.set_data(x[0:showPos+1],y_test[0:showPos+1])
        lineRecordTrain.set_data(x[0:showPos+1],y_train[0:showPos+1])

        pointRecordTest.set_data(x[0:showPos], y_test[0:showPos])
        pointRecordTest.set_marker('.')
        pointRecordTest.set_markersize(4)
        pointRecordTest.set_color('g')

        pointRecordTrain.set_data(x[0:showPos], y_train[0:showPos])
        pointRecordTrain.set_marker('.')
        pointRecordTrain.set_markersize(4)
        pointRecordTrain.set_color('r')

        pointLatestTest.set_data([x[showPos]], [y_test[showPos]])
        pointLatestTest.set_marker(pointMarker[time%len(pointMarker)])
        pointLatestTest.set_markersize(pointSize[time%len(pointSize)])
        pointLatestTest.set_color(pointColor[time%len(pointColor)])

        pointLatestTrain.set_data([x[showPos]], [y_train[showPos]])
        pointLatestTrain.set_marker(pointMarker[(time+1)%len(pointMarker)])
        pointLatestTrain.set_markersize(pointSize[(time+1)%len(pointSize)])
        pointLatestTrain.set_color(pointColor[(time+1)%len(pointColor)])

        return lineRecordTest,lineRecordTrain,lineUp,pointRecordTest,pointRecordTrain,pointLatestTest,pointLatestTrain

    antiRecycle=FuncAnimation(fig,update,frames=range(100),interval=100,blit=True,repeat=True)

    plt.tight_layout()
    plt.show()

def changeToTimeSequence(dataSet:dataSet,numOfTime:int,width:int=0,step:int=1):
    if width==0:
        tempShape=dataSet.x.shape
        dataSet.x=dataSet.x.reshape([tempShape[0],numOfTime,tempShape[2],tempShape[3],tempShape[4],tempShape[5]//numOfTime])
        dataSet.y=torch.repeat_interleave(dataSet.y,numOfTime,1)
    else:
        tempShape=dataSet.x.shape
        tempX=torch.zeros([tempShape[0],numOfTime,tempShape[2],tempShape[3],tempShape[4],width],dtype=torch.float)
        for n in range(tempShape[0]):
            for t in range(numOfTime):
                tempX[n,t,0,0,:,:]=dataSet.x[n,0,0,0,:,t*step:t*step+width].clone()
        dataSet.x=tempX
        dataSet.y=torch.repeat_interleave(dataSet.y,numOfTime,1)

class _modelStructureJsonEncoder(json.JSONEncoder):
    def default(self,obj:Any)->Any:
        if isinstance(obj,artificial_neural_network.actFunc):
            return {'__enum__':'actFunc','__value__':obj.name}
        if isinstance(obj,artificial_neural_network.lossFunc):
            return {'__enum__':'lossFunc','__value__':obj.name}
        if isinstance(obj,artificial_neural_network.typeOfLayer):
            return {'__enum__':'typeOfLayer','__value__':obj.name}
        if isinstance(obj,artificial_neural_network.poolMethod):
            return {'__enum__':'poolMethod','__value__':obj.name}
        if isinstance(obj,artificial_neural_network.recurrentLinkMode):
            return {'__enum__':'recurrentLinkMode','__value__':obj.name}
        if isinstance(obj,artificial_neural_network.recurrentTimeWeightModel):
            return {'__enum__':'recurrentTimeWeightModel','__value__':obj.name}
        if isinstance(obj,artificial_neural_network.timeDirection):
            return {'__enum__':'timeDirection','__value__':obj.name}

        if isinstance(obj,set):
            temp=[]
            for i in obj:
                temp.append(i)
            return {'__annSet__':temp}

        if isinstance(obj,artificial_neural_network.layerInfo):
            tempLyInfo:artificial_neural_network.layerInfo=obj
            return {"typeOfLayer":tempLyInfo.typeOfLayer,
                    "kernelSize":tempLyInfo.kernelSize,
                    "channelSize":tempLyInfo.channelSize,
                    "actFunc":tempLyInfo.actFunc,
                    "poolMethod":tempLyInfo.poolMethod,
                    "channelTable":tempLyInfo.channelTable,
                    "shortCircuitAddSet":tempLyInfo.shortCircuitAddSet,
                    "shortCircuitConcateSet":tempLyInfo.shortCircuitConcateSet,
                    "isRecurrent":tempLyInfo.isRecurrent,
                    "recurrentLinkMode":tempLyInfo.recurrentLinkMode}

        if isinstance(obj, artificial_neural_network.ANN):
            tempANN:artificial_neural_network.ANN=obj
            return {"modelName":tempANN.modelName,
                    "batchSize":tempANN.batchSize,
                    "numOfMaxRecurrent":tempANN.numOfMaxRecurrent,
                    "recurrentTimeWeightModel":tempANN.recurrentTimeWeightModel,
                    "isBidirectional":tempANN.isBidirectional,
                    "lossFunc":tempANN.lossFunc,
                    "layerInfo":tempANN.layerInfo}

        return super().default(obj)

def _enumObjectHook(obj:Any)->Any:
    strToEnum={'actFunc':lambda tempStr:artificial_neural_network.actFunc[tempStr],
               'lossFunc':lambda tempStr:artificial_neural_network.lossFunc[tempStr],
               'typeOfLayer':lambda tempStr:artificial_neural_network.typeOfLayer[tempStr],
               'poolMethod':lambda tempStr:artificial_neural_network.poolMethod[tempStr],
               'recurrentLinkMode':lambda tempStr:artificial_neural_network.recurrentLinkMode[tempStr],
               'recurrentTimeWeightModel':lambda tempStr:artificial_neural_network.recurrentTimeWeightModel[tempStr],
               'timeDirection':lambda tempStr:artificial_neural_network.timeDirection[tempStr]}
    if '__enum__' in obj:
        return strToEnum[obj['__enum__']](obj['__value__'])
    if '__annSet__' in obj:
        temp=set()
        tempList:List=obj['__annSet__']
        for i in range(len(tempList)):
            temp.add(tempList[i])
        return temp

    return obj

def modelStructureSaveToJson(model:ANN,savePath:str=""):
    path:str=os.sep+model.modelName+'-structure-'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.json'
    if savePath=="":
        path=os.path.join(os.path.expanduser("~"),path)
    else:
        path=os.path.join(savePath,path)
    with open(path,'wt',encoding='utf-8') as file: 
        tempJson:str=json.dumps(model,cls=_modelStructureJsonEncoder)
        json.dump(tempJson,file,indent=4,ensure_ascii=False)
        print('model structure success save to: '+path)

def modelParameterSaveToFile(model:ANN,savePath:str=""):
    path:str=os.sep+model.modelName+'-parameter-'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.pth'
    if savePath=="":
        path=os.path.join(os.path.expanduser("~"),path)
    else:
        path=os.path.join(savePath,path)

    para=dict()
    for i in range(len(model.modelPara)):
        para[i]=model.modelPara[i].value.clone()

    count:int=len(model.modelPara)-1
    for i in range(len(model.bnMeanValue)):
        for k in range(3):
            count+=1
            para[count]=model.bnMvEma[k][i].clone()
            count+=1
            para[count]=model.bnDeEma[k][i].clone()
        for j in range(model.numOfMaxRecurrent):
            for k in range(3):
                count+=1
                para[count]=model.bnMvEmaPerTime[j][k][i].clone()
                count+=1
                para[count]=model.bnDeEmaPerTime[j][k][i].clone()

    torch.save(para,path)
    print('model parameter success save to: '+path)

def createModelStructureFromJson(jsonStr:str=None,file:str=None)->artificial_neural_network.ANN:
    model:artificial_neural_network.ANN=None
    modelInfo:Dict=None
    if jsonStr!=None:
        modelInfo=json.loads(jsonStr,object_hook=_enumObjectHook)
    else:
        if file!=None:
            with open(file,'r',encoding='utf-8') as jsonFile:
                modelInfo=json.loads(json.load(jsonFile),object_hook=_enumObjectHook)
    if modelInfo!=None:
        model=artificial_neural_network.ANN(modelName=modelInfo["modelName"],\
                                            batchSize=modelInfo["batchSize"],\
                                            numOfMaxRecurrent=modelInfo["numOfMaxRecurrent"],\
                                            recurrentTimeWeightModel=modelInfo["recurrentTimeWeightModel"],\
                                            isBidirectional=modelInfo["isBidirectional"])
        layInfo:List[artificial_neural_network.layerInfo]=[]
        for tempInfo in modelInfo["layerInfo"]:
            temp=artificial_neural_network.layerInfo(layerType=tempInfo['typeOfLayer'],\
                                                     kernelSize=tempInfo['kernelSize'],\
                                                     channelSize=tempInfo['channelSize'],\
                                                     aFunc=tempInfo['actFunc'],\
                                                     pMethod=tempInfo['poolMethod'],\
                                                     channelTable=tempInfo['channelTable'],\
                                                     shortCircuitAddSet=tempInfo['shortCircuitAddSet'],\
                                                     shortCircuitConcateSet=tempInfo['shortCircuitConcateSet'],\
                                                     isRecurrent=tempInfo['isRecurrent'],\
                                                     rcLinkMode=tempInfo['recurrentLinkMode'])
            layInfo.append(temp)
        model.createAnn(lyInfo=layInfo,lsFunc=modelInfo['lossFunc'],isCheckShare=False)
    return model

def loadModelParameterFromFile(model:artificial_neural_network.ANN,file:str):
    para:Dict=torch.load(file)
    for i in range(len(model.modelPara)):
        model.modelPara[i].value=para[i]
    count:int=len(model.modelPara)-1
    for i in range(len(model.bnMeanValue)):
        for k in range(3):
            count+=1
            model.bnMvEma[k][i]=para[count]
            count+=1
            model.bnDeEma[k][i]=para[count]
        for j in range(model.numOfMaxRecurrent):
            for k in range(3):
                count+=1
                model.bnMvEmaPerTime[j][k][i]=para[count]
                count+=1
                model.bnDeEmaPerTime[j][k][i]=para[count]