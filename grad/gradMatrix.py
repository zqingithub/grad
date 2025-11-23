import copy
from pickle import TRUE
from tkinter import NO
import torch
import grad
import copy

class gradMatrix:

    def __init__(self,value=None,type=grad.Type.num):
        self.left=None
        self.right=None
        self.value=value
        self.type=type
        self.oper=grad.oper.add
        self.grad=None
        self.numOfShare=0
        self.counterCompute=0
        self.counterGrad=0
        self.sumOfGrad=None
        if self.type==grad.Type.gradVariable:
            self.grad=torch.zeros(value.shape)



    def __add__(self,other):
        self.__isGradMatrix(other)
        self.__isShapeMatch(other)
        outcome=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value+other.value
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.add
        outcome.__doShare()
        return outcome


    def __sub__(self,other):
        self.__isGradMatrix(other)
        self.__isShapeMatch(other)
        outcome=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value-other.value
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.sub
        outcome.__doShare()
        return outcome

    def __mul__(self,other):
        self.__isGradMatrix(other)
        self.__isShapeMatch(other)
        outcome=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value*other.value
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.mul
        outcome.__doShare()
        return outcome

    def __truediv__(self,other):
        self.__isGradMatrix(other)
        self.__isShapeMatch(other)
        outcome=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value/other.value
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.div
        outcome.__doShare()
        return outcome

    def __matmul__(self,other):
        self.__isGradMatrix(other)
        if self.value.shape[0:-2]!=other.value.shape[0:-2]:
            raise TypeError("tensor shape unmatch")
        outcome=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=self.value@other.value
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.matmul
        outcome.__doShare()
        return outcome

    def transpose(self,dim1=0,dim2=1):
        outcome=gradMatrix()
        outcome.left=self
        outcome.dim1=dim1
        outcome.dim2=dim2
        outcome.value=self.value.transpose(dim1,dim2)
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.transpose
        outcome.__doShare()
        return outcome

    def reshape(self,reDim):
        outcome=gradMatrix()
        outcome.left=self
        outcome.reDim=reDim
        outcome.value=self.value.reshape(reDim)
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.reshape
        outcome.__doShare()
        return outcome

    def exp(self):
        outcome=gradMatrix()
        outcome.left=self
        outcome.value=torch.exp(self.value)
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.exp
        outcome.__doShare()
        return outcome

    def ReLU(self):
        outcome=gradMatrix()
        outcome.left=self
        outcome.value=self.value
        outcome.value[outcome.value<0]=0
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.ReLU
        outcome.__doShare()
        return outcome

    def LReLU(self):
        outcome=gradMatrix()
        outcome.left=self
        outcome.value=self.value
        outcome.value[outcome.value<0]=outcome.value[outcome.value<0]*0.01
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.LReLU
        outcome.__doShare()
        return outcome

    def abs(self):
        outcome=gradMatrix()
        outcome.left=self
        outcome.value=torch.abs(self.value)
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.abs
        outcome.__doShare()
        return outcome

    def repeat(self,repeatTimes):
        outcome=gradMatrix()
        outcome.left=self
        outcome.value=self.value.repeat(repeatTimes)
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.repeat
        outcome.repeatTimes=repeatTimes
        outcome.__doShare()
        return outcome

    def conv2D(self,other):
        outcome=gradMatrix()
        outcome.left=self
        outcome.right=other
        outcome.value=torch.nn.functional.conv2d(self.value,other.value)
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.conv2D
        outcome.__doShare()
        return outcome

    def maxPool2D(self,kernelSize):
        outcome=gradMatrix()
        outcome.left=self
        outcome.value,outcome.index=torch.nn.MaxPool2d(kernel_size=kernelSize,return_indices=True)(self.value)
        outcome.kernelSize=kernelSize
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.maxPool2D
        outcome.__doShare()
        return outcome

    def avgPool2D(self,kernelSize):
        outcome=gradMatrix()
        outcome.left=self
        outcome.value=torch.nn.AvgPool2d(kernel_size=kernelSize)(self.value)
        outcome.kernelSize=kernelSize
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.avgPool2D
        outcome.__doShare()
        return outcome

    def __sumBySplit(self,ts,dimNo):
        size=len(self.repeatTimes)-len(self.left.value.shape)
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
    def cat(cls,gm1,gm2,dim=0):
        if not (isinstance(gm1,gradMatrix) and isinstance(gm2,gradMatrix)):
            raise TypeError("parameter type is not gradMatrix")
        outcome=gradMatrix()
        outcome.left=gm1
        outcome.right=gm2
        outcome.value=torch.cat([gm1.value,gm2.value],dim)
        outcome.type=grad.Type.midVariable
        outcome.oper=grad.oper.cat
        outcome.dim=dim
        outcome.__doShare()
        return outcome

    @classmethod
    def copy(cls,graph,isShareGradVariable=True,twinsDict={}):
        alreadyCreateDict={}
        return gradMatrix.__copy(graph,alreadyCreateDict,isShareGradVariable,twinsDict)

    @classmethod
    def __copy(cls,graph,alreadyCreateDict,isShareGradVariable,twinsDict):
        if graph==None:
            return None
        if graph in alreadyCreateDict:
            return alreadyCreateDict[graph]
        else:
            if graph.type==grad.Type.gradVariable and isShareGradVariable:
                if graph in twinsDict:
                    twinsDict[graph]=graph
                return graph
            else:
                outcome=copy.copy(graph)
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
        if self.type==grad.Type.midVariable:
            temp=self.numOfShare
            self.numOfShare=1
            self.__computeValue()
            self.numOfShare=temp
        


    def __computeValue(self):
        if self.type==grad.Type.midVariable:
            alreadyCompute=True
            if self.counterCompute==0:
                alreadyCompute=False
            self.counterCompute=(self.counterCompute+1)%self.numOfShare
            if alreadyCompute:
                return
            self.left.__computeValue()
            if self.right!=None:
                self.right.__computeValue()
            if self.oper==grad.oper.add:
                self.value=self.left.value+self.right.value
            if self.oper==grad.oper.sub:
                self.value=self.left.value-self.right.value
            if self.oper==grad.oper.mul:
                self.value=self.left.value*self.right.value
            if self.oper==grad.oper.div:
                self.value=self.left.value/self.right.value
            if self.oper==grad.oper.matmul:
                self.value=self.left.value@self.right.value
            if self.oper==grad.oper.transpose:
                self.value=self.left.value.transpose(self.dim1,self.dim2)
            if self.oper==grad.oper.reshape:
                self.value=self.left.value.reshape(self.reDim)
            if self.oper==grad.oper.repeat:
                self.value=self.left.value.repeat(self.repeatTimes)
            if self.oper==grad.oper.cat:
                self.value=torch.cat([self.left.value,self.right.value],self.dim)
            if self.oper==grad.oper.exp:
                self.value=torch.exp(self.left.value)
            if self.oper==grad.oper.ReLU:
                self.value=self.left.value
                self.value[self.value<0]=0
            if self.oper==grad.oper.LReLU:
                self.value=self.left.value
                self.value[self.value<0]=self.value[self.value<0]*0.01
            if self.oper==grad.oper.abs:
                self.value=torch.abs(self.left.value)
            if self.oper==grad.oper.conv2D:
                self.value=torch.nn.functional.conv2d(self.left.value,self.right.value)
            if self.oper==grad.oper.maxPool2D:
                self.value,self.index=torch.nn.MaxPool2d(kernel_size=self.kernelSize,return_indices=True)(self.left.value)
            if self.oper==grad.oper.avgPool2D:
                self.value=torch.nn.AvgPool2d(kernel_size=self.kernelSize)(self.left.value)

    def __computeGrad(self,preGradValue):
        if self.type==grad.Type.midVariable:
            self.counterGrad+=1
            self.sumOfGrad+=preGradValue
            if self.counterGrad<self.numOfShare:
                return
            else:
                preGradValue=self.sumOfGrad.clone()
                self.sumOfGrad.zero_()
                self.counterGrad=0
            if self.oper==grad.oper.add:
                self.left.__computeGrad(preGradValue)
                self.right.__computeGrad(preGradValue)
            if self.oper==grad.oper.sub:
                self.left.__computeGrad(preGradValue)
                self.right.__computeGrad(preGradValue*(-1.0))
            if self.oper==grad.oper.mul:
                self.left.__computeGrad(preGradValue*self.right.value)
                self.right.__computeGrad(preGradValue*self.left.value)
            if self.oper==grad.oper.div:
                self.left.__computeGrad(preGradValue*(1.0/self.right.value))
                self.right.__computeGrad(preGradValue*self.left.value*(-1.0/(self.right.value*self.right.value)))
            if self.oper==grad.oper.matmul:
                self.left.__computeGrad(preGradValue@self.right.value.transpose(-1,-2))
                self.right.__computeGrad(self.left.value.transpose(-1,-2)@preGradValue)
            if self.oper==grad.oper.transpose:
                self.left.__computeGrad(preGradValue.transpose(self.dim1,self.dim2))
            if self.oper==grad.oper.reshape:
                self.left.__computeGrad(preGradValue.reshape(self.left.value.shape))
            if self.oper==grad.oper.repeat:
                self.left.__computeGrad(self.__sumBySplit(preGradValue,0))
            if self.oper==grad.oper.cat:
                index=self.left.value.shape[self.dim]
                lpr,rpr=torch.split(preGradValue,[index,self.value.shape[self.dim]-index],self.dim)
                self.left.__computeGrad(lpr)
                self.right.__computeGrad(rpr)
            if self.oper==grad.oper.exp:
                self.left.__computeGrad(preGradValue*self.value)
            if self.oper==grad.oper.ReLU:
                nextGradValue=preGradValue
                nextGradValue[self.value<=0]=0
                self.left.__computeGrad(nextGradValue)
            if self.oper==grad.oper.LReLU:
                nextGradValue=preGradValue
                nextGradValue[self.value<=0]=nextGradValue[self.value<=0]*0.01
                self.left.__computeGrad(nextGradValue)
            if self.oper==grad.oper.abs:
                nextGradValue=preGradValue
                index=self.left.value<0
                nextGradValue[index]=-nextGradValue[index]
                self.left.__computeGrad(nextGradValue)
            if self.oper==grad.oper.conv2D:

                nextGradValue=torch.zeros(self.left.value.shape,dtype=torch.float)
                for i in range(self.right.value.shape[0]):
                    temp=preGradValue[0,i].unsqueeze(0).unsqueeze(0)
                    kernel=preGradValue[0,i].unsqueeze(0).unsqueeze(0).repeat([self.left.value.shape[1],1,1,1])
                    nextGradValue+=torch.nn.functional.conv2d(torch.flip(self.right.value[i],(1,2)).unsqueeze(0),kernel,\
                        groups=self.left.value.shape[1],padding=(self.left.value.shape[-2]-self.right.value.shape[-2],\
                                                                 self.left.value.shape[-1]-self.right.value.shape[-1]))
                self.left.__computeGrad(torch.flip(nextGradValue,(-1,-2)))

                nextGradValue=torch.zeros(self.right.value.shape,dtype=torch.float)
                for i in range(self.right.value.shape[0]):
                    kernel=preGradValue[0,i].unsqueeze(0).unsqueeze(0).repeat([self.left.value.shape[1],1,1,1])
                    nextGradValue[i]=torch.nn.functional.conv2d(self.left.value,kernel,groups=self.left.value.shape[1])[0]
                self.right.__computeGrad(nextGradValue)
                
            if self.oper==grad.oper.maxPool2D:
                nextGradValue=torch.zeros(self.left.value.shape,dtype=torch.float)
                listSize=self.index.shape[-1]*self.index.shape[-2]
                listSize2=nextGradValue.shape[-1]*nextGradValue.shape[-2]
                for i in range(nextGradValue.shape[0]):
                    for j in range(nextGradValue.shape[1]):
                        colIndex=self.index[i,j].reshape(1,listSize)
                        rowIndex=torch.zeros(1,listSize,dtype=torch.long)
                        temp=nextGradValue[i,j].reshape(1,listSize2)
                        temp.index_put_((rowIndex,colIndex),preGradValue[i,j].reshape(1,listSize))
                        nextGradValue[i,j]=temp.reshape(nextGradValue.shape[-2],nextGradValue.shape[-1])
                self.left.__computeGrad(nextGradValue)

            if self.oper==grad.oper.avgPool2D:
                nextGradValue=torch.repeat_interleave(preGradValue,self.kernelSize,-1)
                nextGradValue=torch.repeat_interleave(nextGradValue,self.kernelSize,-2)/(self.kernelSize*self.kernelSize)
                self.left.__computeGrad(torch.nn.functional.pad(nextGradValue,(0,self.left.value.shape[-1]\
                    -nextGradValue.shape[-1],0,self.left.value.shape[-2]-nextGradValue.shape[-2])))

        if self.type==grad.Type.gradVariable:
            self.grad=self.grad+preGradValue

    def __isShapeMatch(self,other):
        if self.value.shape!=other.value.shape:
            raise TypeError("tensor shape unmatch")

    def __isGradMatrix(self,other):
        if not isinstance(other,gradMatrix):
            raise TypeError("parameter type is not gradMatrix")

    def __doShare(self):
        self.sumOfGrad=torch.zeros(self.value.shape)
        if self.left.type==grad.Type.midVariable:
                self.left.numOfShare+=1
        if self.right!=None and self.right.type==grad.Type.midVariable:
                self.right.numOfShare+=1


    def computeGrad(self,gradVariable,preGradValue=None):
        for i in gradVariable:
            i.grad.zero_()
        if preGradValue==None:
            preGradValue=torch.ones(self.value.shape,dtype=torch.float)
        temp=self.numOfShare
        self.numOfShare=1
        self.__computeGrad(preGradValue)
        self.numOfShare=temp

  
    def computeGradByEstimate(self,gradVariable,step):
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