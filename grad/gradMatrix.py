#在构建神经网络的过程中，同时构建计算图，后续通过计算图可以自动求模型参数的梯度，自动求梯度的方法采用自上向下的方法
from __future__ import annotations
from ast import Dict, List
import copy
import math
import torch
from enum import Enum

class variableType(Enum):
    """
    用于定义计算图中节点的类型：
    num：表示节点用于接收模型的非参数输入
    midVaraiable:表示计算图生成过程中的中间变量
    gradVariable:表示模型的参数,后续会计算该节点的梯度
    """
    num=0
    midVariable=1
    gradVariable=2

class operatorType(Enum):
    """
    用于表示计算图中间变量的操作（或计算）的类型
    """
    add=0
    sub=1
    mul=2
    div=3
    sigmoid=4
    pow=5
    matmul=6
    transpose=7
    repeat=8
    cat=9
    exp=10
    ReLU=11
    abs=12
    conv2D=13
    maxPool2D=14
    avgPool2D=15
    reshape=16
    LReLU=17
    max=18
    softmax=19
    log=20
    index_select=21



class gradMatrix:

    def __init__(self,value:torch.tensor=None,type:variableType=variableType.num):
        """
        用于表示计算图中的一个节点
        left：节点对应操作的第一个操作数（也用gradMatrix类对象表示）
        right：节点对应操作的第二个操作数（也用gradMatrix类对象表示）
        value：用于记录节点在计算图计算过程中的计算值
        type：节点的类型
        oper：节点对应操作的类型
        grad：假如节点表示的是模型的参数，则用grad记录参数的梯度
        numOfShare：用于记录有多少个节点把当前的节点用于计算（即有操作数指针指向当前节点），用于避免计算图计算或参数求梯度过程中的重复计算
        counterCompute：用于记录在计算图计算过程中节点被请求计算的次数，用于避免重复计算，当counterCompute大于0时表示已被请求过，
                        则直接返回value记录的计算值，当和numOfShare一样时，则表示所有把当前节点当作操作数的节点已全部请求过当前节点的计算值，
                        可以重置计算结果。
        counterGrad：作用和counterCompute类似，表示在求模型参数梯度的过程中，有多少个把当前节点当作操作数的节点已将梯度累加至sumOfGrad（计算当前节点梯度时需要用到的由上传递下来的梯度）
                     中，当和numOfShare一样时，则表示所有把当前节点当作操作数的节点已全部完成梯度的计算，当前节点在计算梯度过程中需要用到的sumOfGrad值已计算完成，可以开始当前节点的梯度计算
        sumOfGrad：用于累加所有把当前节点当作操作数的节点传递过来的梯度
        """
        self.left:gradMatrix=None
        self.right:gradMatrix=None
        self.value:torch.tensor=value
        self.type:variableType=type
        self.oper:operatorType=operatorType.add
        self.grad:torch.tensor=None
        self.numOfShare:int=0
        self.counterCompute:int=0
        self.counterGrad:int=0
        self.sumOfGrad:torch.tensor=None
        self.dim1:int=-1
        self.dim2:int=-1
        self.reDim:list[int]=None
        self.repeatTimes:list[int]=None
        self.kernelSize:int=0
        self.indexMaxPool:torch.tensor=None
        self.dim:int=-1
        self.exponent:float=0
        if self.type==variableType.gradVariable:
            self.grad=torch.zeros(value.shape)



    def __add__(self,other:gradMatrix)->gradMatrix:
        self.__isGradMatrix(other)
        self.__isShapeMatch(other)
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value+other.value
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.add
        outcome.__doShare()
        return outcome


    def __sub__(self,other:gradMatrix)->gradMatrix:
        self.__isGradMatrix(other)
        self.__isShapeMatch(other)
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value-other.value
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.sub
        outcome.__doShare()
        return outcome

    def __mul__(self,other:gradMatrix)->gradMatrix:
        self.__isGradMatrix(other)
        self.__isShapeMatch(other)
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value*other.value
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.mul
        outcome.__doShare()
        return outcome

    def __truediv__(self,other:gradMatrix)->gradMatrix:
        self.__isGradMatrix(other)
        self.__isShapeMatch(other)
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value/other.value
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.div
        outcome.__doShare()
        return outcome

    def pow(self,exponent:float)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.exponent=exponent
        outcome.value=torch.pow(self.value,exponent)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.pow
        outcome.__doShare()
        return outcome

    def __matmul__(self,other:gradMatrix)->gradMatrix:
        self.__isGradMatrix(other)
        if self.value.shape[0:-2]!=other.value.shape[0:-2]:
            raise TypeError("tensor shape unmatch")
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value@other.value
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.matmul
        outcome.__doShare()
        return outcome

    def transpose(self,dim1:int=0,dim2:int=1)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.dim1=dim1
        outcome.dim2=dim2
        outcome.value=self.value.transpose(dim1,dim2)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.transpose
        outcome.__doShare()
        return outcome

    def reshape(self,reDim:list[int])->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.reDim=reDim
        outcome.value=self.value.reshape(reDim)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.reshape
        outcome.__doShare()
        return outcome

    def exp(self)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value=torch.exp(self.value)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.exp
        outcome.__doShare()
        return outcome

    def log(self)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value=torch.log(self.value)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.log
        outcome.__doShare()
        return outcome

    def ReLU(self)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value=self.value
        outcome.value[outcome.value<0]=0
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.ReLU
        outcome.__doShare()
        return outcome

    def LReLU(self)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value=self.value
        outcome.value[outcome.value<0]=outcome.value[outcome.value<0]*0.01
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.LReLU
        outcome.__doShare()
        return outcome

    def sigmoid(self)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        temp1=self.value/2.0
        maxValue=torch.abs(temp1)
        temp2=torch.exp(-temp1-maxValue)
        temp1=torch.exp(temp1-maxValue)
        outcome.value=temp1/(temp1+temp2)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.sigmoid
        outcome.__doShare()
        return outcome

    def softmax(self)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        maxValue=torch.max(self.value)
        temp=torch.exp(self.value-maxValue)
        sum=torch.sum(temp)
        outcome.value=temp/sum
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.softmax
        outcome.__doShare()
        return outcome

    def abs(self)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value=torch.abs(self.value)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.abs
        outcome.__doShare()
        return outcome

    def max(self)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value=torch.max(self.value).reshape(1,1)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.max
        outcome.__doShare()
        return outcome

    def repeat(self,repeatTimes:List[int])->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value=self.value.repeat(repeatTimes)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.repeat
        outcome.repeatTimes=repeatTimes
        outcome.__doShare()
        return outcome

    def index_select(self,dim:int,index:torch.tensor)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self

        temp1,temp2=torch.sort(index)
        for i in range(1,temp1.shape[0]):
            if temp1[i]==temp1[i-1]:
                raise TypeError("index_select oper has same index")
        if temp1[-1]>=self.value.shape[dim]:
            raise TypeError("index_select oper index overstep the bound")

        outcome.value=torch.index_select(self.value,dim,index)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.index_select
        outcome.dim=dim
        outcome.indexMaxPool=index
        outcome.__doShare()
        return outcome

    def conv2D(self,other:gradMatrix)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=torch.nn.functional.conv2d(self.value,other.value,padding=((other.value.shape[2]-1)//2,(other.value.shape[3]-1)//2))
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.conv2D
        outcome.__doShare()
        return outcome

    def maxPool2D(self,kernelSize:int)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value,outcome.indexMaxPool=torch.nn.MaxPool2d(kernel_size=kernelSize,return_indices=True)(self.value)
        outcome.kernelSize=kernelSize
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.maxPool2D
        outcome.__doShare()
        return outcome

    def avgPool2D(self,kernelSize:int)->gradMatrix:
        outcome:gradMatrix=gradMatrix()
        outcome.left=self
        outcome.value=torch.nn.AvgPool2d(kernel_size=kernelSize)(self.value)
        outcome.kernelSize=kernelSize
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.avgPool2D
        outcome.__doShare()
        return outcome

    def __sumBySplit(self,ts:torch.tensor,dimNo:int)->torch.tensor:
        size:int=len(self.repeatTimes)-len(self.left.value.shape)
        if dimNo==len(self.repeatTimes):
            return ts
        else:
            sumGrad=torch.zeros(self.left.value.shape)
            if dimNo<size:
                tsList=torch.split(ts,ts.shape[0]//self.repeatTimes[dimNo],0)
                for i in tsList:
                    sumGrad=sumGrad+self.__sumBySplit(i.squeeze(0),dimNo+1)
            else:
                tsList=torch.split(ts,ts.shape[dimNo-size]//self.repeatTimes[dimNo],dimNo-size)
                for i in tsList:
                    sumGrad=sumGrad+self.__sumBySplit(i,dimNo+1)
            return sumGrad



    @classmethod
    def cat(cls,gm1:gradMatrix,gm2:gradMatrix,dim:int=0)->gradMatrix:
        if not (isinstance(gm1,gradMatrix) and isinstance(gm2,gradMatrix)):
            raise TypeError("parameter type is not gradMatrix")
        outcome:gradMatrix=gradMatrix()
        outcome.left=gm1
        outcome.right=gm2
        outcome.value=torch.cat([gm1.value,gm2.value],dim)
        outcome.type=variableType.midVariable
        outcome.oper=operatorType.cat
        outcome.dim=dim
        outcome.__doShare()
        return outcome

    @classmethod
    def copy(cls,graph:gradMatrix,isShareGradVariable:bool=True,twinsDict:Dict['gradMatrix','gradMatrix']={})->gradMatrix:
        alreadyCreateDict:Dict['gradMatrix','gradMatrix']={}
        return gradMatrix.__copy(graph,alreadyCreateDict,isShareGradVariable,twinsDict)

    @classmethod
    def __copy(cls,graph:gradMatrix,alreadyCreateDict:Dict['gradMatrix','gradMatrix'],isShareGradVariable:bool,twinsDict:Dict['gradMatrix','gradMatrix'])->gradMatrix:
        if graph==None:
            return None
        if graph in alreadyCreateDict:
            return alreadyCreateDict[graph]
        else:
            if graph.type==variableType.gradVariable and isShareGradVariable:
                if graph in twinsDict:
                    twinsDict[graph]=graph
                return graph
            else:
                outcome:gradMatrix=copy.copy(graph)
                outcome.value=graph.value.clone()
                outcome.grad=graph.grad.clone() if graph.grad != None else None
                outcome.sumOfGrad=graph.sumOfGrad.clone() if graph.sumOfGrad != None else None
                outcome.left=gradMatrix.__copy(graph.left,alreadyCreateDict,isShareGradVariable,twinsDict)
                outcome.right=gradMatrix.__copy(graph.right,alreadyCreateDict,isShareGradVariable,twinsDict)
                alreadyCreateDict[graph]=outcome
                if graph in twinsDict:
                    twinsDict[graph]=outcome
                return outcome

    def computeValue(self):
        if self.type==variableType.midVariable:
            temp:int=self.numOfShare
            self.numOfShare=1
            self.__computeValue()
            self.numOfShare=temp
        


    def __computeValue(self):
        if self.type==variableType.midVariable:
            alreadyCompute:bool=True
            if self.counterCompute==0:
                alreadyCompute=False
            self.counterCompute=(self.counterCompute+1)%self.numOfShare
            if alreadyCompute:
                return
            self.left.__computeValue()
            if self.right!=None:
                self.right.__computeValue()
            if self.oper==operatorType.add:
                self.value=self.left.value+self.right.value
                return
            if self.oper==operatorType.sub:
                self.value=self.left.value-self.right.value
                return
            if self.oper==operatorType.mul:
                self.value=self.left.value*self.right.value
                return
            if self.oper==operatorType.div:
                self.value=self.left.value/self.right.value
                return
            if self.oper==operatorType.pow:
                self.value=torch.pow(self.left.value,self.exponent)
                return
            if self.oper==operatorType.matmul:
                self.value=self.left.value@self.right.value
                return
            if self.oper==operatorType.transpose:
                self.value=self.left.value.transpose(self.dim1,self.dim2)
                return
            if self.oper==operatorType.reshape:
                self.value=self.left.value.reshape(self.reDim)
                return
            if self.oper==operatorType.repeat:
                self.value=self.left.value.repeat(self.repeatTimes)
                return
            if self.oper==operatorType.index_select:
                self.value=torch.index_select(self.left.value,self.dim,self.indexMaxPool)
                return
            if self.oper==operatorType.cat:
                self.value=torch.cat([self.left.value,self.right.value],self.dim)
                return
            if self.oper==operatorType.exp:
                self.value=torch.exp(self.left.value)
                return
            if self.oper==operatorType.log:
                self.value=torch.log(self.left.value)
                return
            if self.oper==operatorType.ReLU:
                self.value=self.left.value
                self.value[self.value<0]=0
                return
            if self.oper==operatorType.LReLU:
                self.value=self.left.value
                self.value[self.value<0]=self.value[self.value<0]*0.01
                return
            if self.oper==operatorType.sigmoid:
                temp1=self.left.value/2.0
                maxValue=torch.abs(temp1)
                temp2=torch.exp(-temp1-maxValue)
                temp1=torch.exp(temp1-maxValue)
                self.value=temp1/(temp1+temp2)
                return
            if self.oper==operatorType.softmax:
                maxValue=torch.max(self.left.value)
                temp=torch.exp(self.left.value-maxValue)
                sum=torch.sum(temp)
                self.value=temp/sum
                return
            if self.oper==operatorType.abs:
                self.value=torch.abs(self.left.value)
                return
            if self.oper==operatorType.max:
                self.value=torch.max(self.left.value).reshape(1,1)
                return
            if self.oper==operatorType.conv2D:
                self.value=torch.nn.functional.conv2d(self.left.value,self.right.value,padding=((self.right.value.shape[2]-1)//2,(self.right.value.shape[3]-1)//2))
                return
            if self.oper==operatorType.maxPool2D:
                self.value,self.indexMaxPool=torch.nn.MaxPool2d(kernel_size=self.kernelSize,return_indices=True)(self.left.value)
                return
            if self.oper==operatorType.avgPool2D:
                self.value=torch.nn.AvgPool2d(kernel_size=self.kernelSize)(self.left.value)
                return

    def __computeGrad(self,preGradValue:torch.tensor):
        if self.type==variableType.midVariable:
            self.counterGrad+=1
            self.sumOfGrad+=preGradValue
            if self.counterGrad<self.numOfShare:
                return
            else:
                preGradValue=self.sumOfGrad.clone()
                self.sumOfGrad.zero_()
                self.counterGrad=0
            if self.oper==operatorType.add:
                self.left.__computeGrad(preGradValue)
                self.right.__computeGrad(preGradValue)
                return
            if self.oper==operatorType.sub:
                self.left.__computeGrad(preGradValue)
                self.right.__computeGrad(preGradValue*(-1.0))
                return
            if self.oper==operatorType.mul:
                self.left.__computeGrad(preGradValue*self.right.value)
                self.right.__computeGrad(preGradValue*self.left.value)
                return
            if self.oper==operatorType.div:
                self.left.__computeGrad(preGradValue*(1.0/self.right.value))
                self.right.__computeGrad(preGradValue*self.left.value*(-1.0/(self.right.value*self.right.value)))
                return
            if self.oper==operatorType.pow:
                nextGradValue=torch.ones(self.left.value.shape,dtype=torch.float)*self.exponent*torch.pow(self.left.value,self.exponent-1)*preGradValue
                self.left.__computeGrad(nextGradValue)  
                return
            if self.oper==operatorType.matmul:
                self.left.__computeGrad(preGradValue@self.right.value.transpose(-1,-2))
                self.right.__computeGrad(self.left.value.transpose(-1,-2)@preGradValue)
                return
            if self.oper==operatorType.transpose:
                self.left.__computeGrad(preGradValue.transpose(self.dim1,self.dim2))
                return
            if self.oper==operatorType.reshape:
                self.left.__computeGrad(preGradValue.reshape(self.left.value.shape))
                return
            if self.oper==operatorType.repeat:
                self.left.__computeGrad(self.__sumBySplit(preGradValue,0))
                return
            if self.oper==operatorType.index_select:
                nextGradValue=torch.zeros(self.left.value.shape,dtype=torch.float)          
                self.left.__computeGrad(nextGradValue.index_copy(self.dim,self.indexMaxPool,preGradValue))
                return
            if self.oper==operatorType.cat:
                index=self.left.value.shape[self.dim]
                lpr,rpr=torch.split(preGradValue,[index,self.value.shape[self.dim]-index],self.dim)
                self.left.__computeGrad(lpr)
                self.right.__computeGrad(rpr)
                return
            if self.oper==operatorType.exp:
                self.left.__computeGrad(preGradValue*self.value)
                return
            if self.oper==operatorType.log:
                self.left.__computeGrad(preGradValue/self.left.value)
                return
            if self.oper==operatorType.ReLU:
                nextGradValue=preGradValue
                nextGradValue[self.value<=0]=0
                self.left.__computeGrad(nextGradValue)
                return
            if self.oper==operatorType.LReLU:
                nextGradValue=preGradValue
                nextGradValue[self.value<=0]=nextGradValue[self.value<=0]*0.01
                self.left.__computeGrad(nextGradValue)
                return
            if self.oper==operatorType.sigmoid:
                self.left.__computeGrad(preGradValue*self.value*(1-self.value))
                return
            if self.oper==operatorType.softmax:
                size=math.prod(preGradValue.shape)
                temp1=preGradValue.reshape([1,size])
                temp2=self.value.reshape([1,size])
                temp3=temp2.repeat([size,1])
                temp4=temp1*temp2
                temp5=-temp4
                temp5=temp5@temp3+temp4
                self.left.__computeGrad(temp5.reshape(preGradValue.shape))
                return
            if self.oper==operatorType.abs:
                nextGradValue=preGradValue
                index=self.left.value<0
                nextGradValue[index]=-nextGradValue[index]
                self.left.__computeGrad(nextGradValue)
                return
            if self.oper==operatorType.max:
                nextGradValue=torch.zeros(self.left.value.shape,dtype=torch.float)
                index=torch.argmax(self.left.value)
                nextGradValue.view(-1)[index]=preGradValue[0,0].item()
                self.left.__computeGrad(nextGradValue)
                return
            if self.oper==operatorType.conv2D:
                nextGradValue=torch.nn.functional.conv2d(preGradValue,torch.flip(self.right.value.transpose(0,1),(-1,-2)),padding=((self.right.value.shape[2]-1)//2,(self.right.value.shape[3]-1)//2))
                self.left.__computeGrad(nextGradValue)
                nextGradValue=torch.nn.functional.conv2d(self.left.value.transpose(0,1),preGradValue.transpose(0,1),padding=((self.right.value.shape[2]-1)//2,(self.right.value.shape[3]-1)//2))
                self.right.__computeGrad(nextGradValue.transpose(0,1))   
                return
            if self.oper==operatorType.maxPool2D:
                nextGradValue=torch.zeros(self.left.value.shape,dtype=torch.float)
                listSize=self.indexMaxPool.shape[-1]*self.indexMaxPool.shape[-2]
                listSize2=nextGradValue.shape[-1]*nextGradValue.shape[-2]
                for i in range(nextGradValue.shape[0]):
                    for j in range(nextGradValue.shape[1]):
                        colIndex=self.indexMaxPool[i,j].reshape(1,listSize)
                        rowIndex=torch.zeros(1,listSize,dtype=torch.long)
                        temp=nextGradValue[i,j].reshape(1,listSize2)
                        temp.index_put_((rowIndex,colIndex),preGradValue[i,j].reshape(1,listSize))
                        nextGradValue[i,j]=temp.reshape(nextGradValue.shape[-2],nextGradValue.shape[-1])
                self.left.__computeGrad(nextGradValue)
                return
            if self.oper==operatorType.avgPool2D:
                nextGradValue=torch.repeat_interleave(preGradValue,self.kernelSize,-1)
                nextGradValue=torch.repeat_interleave(nextGradValue,self.kernelSize,-2)/(self.kernelSize*self.kernelSize)
                self.left.__computeGrad(torch.nn.functional.pad(nextGradValue,(0,self.left.value.shape[-1]\
                    -nextGradValue.shape[-1],0,self.left.value.shape[-2]-nextGradValue.shape[-2])))
                return

        if self.type==variableType.gradVariable:
            self.grad=self.grad+preGradValue

    def __isShapeMatch(self,other:torch.tensor):
        if self.value.shape!=other.value.shape:
            raise TypeError("tensor shape unmatch")

    def __isGradMatrix(self,other:torch.tensor):
        if not isinstance(other,gradMatrix):
            raise TypeError("parameter type is not gradMatrix")

    def __doShare(self):
        self.sumOfGrad=torch.zeros(self.value.shape)
        if self.left.type==variableType.midVariable:
                self.left.numOfShare+=1
        if self.right!=None and self.right.type==variableType.midVariable:
                self.right.numOfShare+=1


    def computeGrad(self,gradVariable:List['gradMatrix'],preGradValue:torch.tensor=None):
        for i in gradVariable:
            i.grad.zero_()
        if preGradValue==None:
            preGradValue=torch.ones(self.value.shape,dtype=torch.float)
        temp:int=self.numOfShare
        self.numOfShare=1
        self.__computeGrad(preGradValue)
        self.numOfShare=temp

  
    def computeGradByEstimate(self,gradVariable:List['gradMatrix'],step:float):
        for g in gradVariable:
            tempValue=g.value.view(-1)
            tempGrad=g.grad.view(-1)
            for i in range(tempValue.shape[0]):
                    gradStore=tempValue[i].item()
                    tempValue[i]=gradStore+step
                    self.computeValue()
                    y1=self.value[0,0].item()
                    tempValue[i]=gradStore-step
                    self.computeValue()
                    y2=self.value[0,0].item()
                    tempValue[i]=gradStore
                    tempGrad[i]=(y1-y2)/2.0/step

    def resetShare(self):
        if self.type==variableType.midVariable:
            self.counterCompute=0
            self.counterGrad=0
            self.sumOfGrad.zero_()
        if self.left!=None:
            self.left.resetShare()
        if self.right!=None:
            self.right.resetShare()

    def __checkShare(self)->bool:
        if self.counterCompute!=0:
            return False
        if self.left!=None:
            if not self.left.__checkShare():
                return False
        if self.right!=None:
            if not self.right.__checkShare():
                return False
        return True

    def checkShare(self):
        self.computeValue()
        if not self.__checkShare():
            raise TypeError("midVariable leakage")
        self.resetShare()