from __future__ import annotations
from ast import List, Tuple
from multiprocessing import Queue
import time
import numpy
import torch.utils.data.dataset
import torchvision
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import random
import threading
from artificial_neural_network import ANN
import math

class dataSet:
    def __init__(self,x:torch.tensor=None,y:torch.tensor=None):
        self.x:torch.tensor=x
        self.y:torch.tensor=y

def getDataSet()->Tuple[dataSet]:
    trainDS:torch.utils.data.dataset=torchvision.datasets.MNIST(root="C:/data",train=True,download=False)
    nSample:int=len(trainDS.targets)
    trainDS.data=trainDS.data.cuda()
    trainDS.targets=trainDS.targets.cuda()
    td:torch.tensor=trainDS.data/torch.ones(trainDS.data.shape,dtype=torch.float)/255
    td=td.unsqueeze(1).unsqueeze(1)
    tl:torch.tensor=torch.zeros((nSample,10),dtype=torch.float)
    tl.scatter_(1,trainDS.targets.unsqueeze(1),torch.ones((nSample,1),dtype=torch.float))
    tl=tl.unsqueeze(1)
    for i in range(nSample):
        pos=random.randint(i,nSample-1)
        ex=td[pos]
        td[pos]=td[i]
        td[i]=ex
        ex=tl[pos]
        tl[pos]=tl[i]
        tl[i]=ex

    DS1:dataSet=dataSet(td,tl)
    testDS:torch.utils.data.dataset=torchvision.datasets.MNIST(root="C:/data",train=False,download=False)
    nSample=len(testDS.targets)
    testDS.data=testDS.data.cuda()
    testDS.targets=testDS.targets.cuda()
    td=testDS.data/torch.ones(testDS.data.shape,dtype=torch.float)/255
    td=td.unsqueeze(1).unsqueeze(1)
    tl=torch.zeros((nSample,10),dtype=torch.float)
    tl.scatter_(1,testDS.targets.unsqueeze(1),torch.ones((nSample,1),dtype=torch.float))
    tl=tl.unsqueeze(1)
    for i in range(nSample):
        pos=random.randint(i,nSample-1)
        ex=td[pos]
        td[pos]=td[i]
        td[i]=ex
        ex=tl[pos]
        tl[pos]=tl[i]
        tl[i]=ex

    return (DS1,dataSet(td,tl))

'''
    figure=pyplot.figure(figsize=(8,8))
    for i in range(100):
        pos=random.randint(0,nSample-1)
        pyplot.title(tl[pos])
        pyplot.imshow(td[pos].reshape(28,28))
        pyplot.show()
'''

def computeAcc(aNN:ANN,testDS:dataSet,extractNum:int=100,isUseFinishModel:bool=False)->float:
    nSample:int=testDS.x.shape[0]
    sum:float=0.0
    pos:int=random.randint(0,nSample-1)
    if not isUseFinishModel:
        ite=math.ceil(extractNum/aNN.batchSize)
        for i in range(ite):
            aNN.setXBatch(testDS.x,pos)
            aNN.computeValue()
            for j in range(aNN.batchSize):
                if torch.argmax(aNN.y[j].value)==torch.argmax(testDS.y[(pos+j)%nSample][0]):
                    sum=sum+1
            pos+=aNN.batchSize
        return sum*1.0/ite/aNN.batchSize
    else:
        for i in range(extractNum):
            tp=(i+pos)%nSample
            if torch.argmax(aNN.getYByFinishModel(testDS.x[tp:tp+1]))==torch.argmax(testDS.y[(i+pos)%nSample][0]):
                sum=sum+1
        return sum*1.0/extractNum


def showTrainProcess(size:int,queue:Queue,modelName:str=""):

    x=range(size)
    y=[0.0]*size
    pos=[1]

    drawLineThread:threading.Thread=threading.Thread(target=drawLine,name='drawLineThread',kwargs={'x':x,'y':y,'pos':pos,'modelName':modelName},daemon=True)
    drawLineThread.start()

    while True:
        try:
            y[pos[0]]=queue.get_nowait()
            if y[pos[0]]<0:
                break
            pos[0]=(pos[0]+1)%size
        except:
            time.sleep(1)

    drawLineThread.terminate()
    

def drawLine(x:List[int],y:List[float],pos:List[int],modelName:str=""):

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

    line,=ax.plot([],[],'g-',linewidth=2)
    pointRecord,=ax.plot(0,0,'ro',markersize=8)
    pointLatest,=ax.plot(0,0,'ro',markersize=8)

    pointMarker=['s','x','o','*']
    pointSize=[4,8,6,10]
    pointColor=['r','g','b','y']

    def update(time):

        showPos=(pos[0]-1)%len(x)
        line.set_data(x[0:showPos+1],y[0:showPos+1])

        pointRecord.set_data(x[0:showPos], y[0:showPos])
        pointRecord.set_marker('.')
        pointRecord.set_markersize(5)
        pointRecord.set_color('k')

        pointLatest.set_data([x[showPos]], [y[showPos]])
        pointLatest.set_marker(pointMarker[time%len(pointMarker)])
        pointLatest.set_markersize(pointSize[time%len(pointSize)])
        pointLatest.set_color(pointColor[time%len(pointColor)])

        return line,pointRecord,pointLatest

    antiRecycle=FuncAnimation(fig,update,frames=range(100),interval=100,blit=True,repeat=True)

    plt.tight_layout()
    plt.show()




