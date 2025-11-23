from enum import Enum
import math

class Type(Enum):
	num=0
	midVariable=1
	gradVariable=2

class oper(Enum):
    add=0
    sub=1
    mul=2
    div=3
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


     
class grad:
    def __init__(self):
        self.left=None
        self.right=None
        self.value=0
        self.Type=Type.num
        self.oper=oper.add
        self.gradVariablePos=-1

    
    def __add__(self,other):
        if isinstance(other,grad):
            outcome=grad()
            outcome.left=self
            outcome.right=other
            outcome.value=self.value+other.value
            outcome.Type=Type.midVariable
            outcome.oper=oper.add
            return outcome
        if isinstance(other,int) or isinstance(other,float):
            return self+grad.createNum(other)
        raise TypeError("Unsupported operand type for +")

    def __sub__(self,other):
        if isinstance(other,grad):
            outcome=grad()
            outcome.left=self
            outcome.right=other
            outcome.value=self.value-other.value
            outcome.Type=Type.midVariable
            outcome.oper=oper.sub
            return outcome
        if isinstance(other,int) or isinstance(other,float):
            return self-grad.createNum(other)
        raise TypeError("Unsupported operand type for -")

    def __mul__(self,other):
        if isinstance(other,grad):
            outcome=grad()
            outcome.left=self
            outcome.right=other
            outcome.value=self.value*other.value
            outcome.Type=Type.midVariable
            outcome.oper=oper.mul
            return outcome
        if isinstance(other,int) or isinstance(other,float):
            return self*grad.createNum(other)
        raise TypeError("Unsupported operand type for *")

    def __truediv__(self,other):
        if isinstance(other,grad):
            outcome=grad()
            outcome.left=self
            outcome.right=other
            outcome.value=self.value/other.value
            outcome.Type=Type.midVariable
            outcome.oper=oper.div
            return outcome
        if isinstance(other,int) or isinstance(other,float):
            return self/grad.createNum(other)
        raise TypeError("Unsupported operand type for /")

    def __pow__(self,other):
        if isinstance(other,grad):
            outcome=grad()
            outcome.left=self
            outcome.right=other
            outcome.value=pow(self.value,other.value)
            outcome.Type=Type.midVariable
            outcome.oper=oper.pow
            return outcome
        if isinstance(other,int) or isinstance(other,float):
            return pow(self,grad.createNum(other))
        raise TypeError("Unsupported operand type for pow")

    def computeValue(self):
        if self.Type==Type.midVariable:
            self.left.computeValue()
            self.right.computeValue()
            if self.oper==oper.add:
                self.value=self.left.value+self.right.value
            if self.oper==oper.sub:
                self.value=self.left.value-self.right.value
            if self.oper==oper.mul:
                self.value=self.left.value*self.right.value
            if self.oper==oper.div:
                self.value=self.left.value/self.right.value
            if self.oper==oper.pow:
                self.value=pow(self.left.value,self.right.value)

    def computeGrad(self,outcome,preGradValue):
        if self.Type==Type.midVariable:
            if self.oper==oper.add:
                self.left.computeGrad(outcome,preGradValue)
                self.right.computeGrad(outcome,preGradValue)
            if self.oper==oper.sub:
                self.left.computeGrad(outcome,preGradValue)
                self.right.computeGrad(outcome,-preGradValue)
            if self.oper==oper.mul:
                self.left.computeGrad(outcome,preGradValue*self.right.value)
                self.right.computeGrad(outcome,preGradValue*self.left.value)
            if self.oper==oper.div:
                self.left.computeGrad(outcome,preGradValue*1.0/self.right.value)
                self.right.computeGrad(outcome,-preGradValue*self.left.value*1.0/self.right.value/self.right.value)
            if self.oper==oper.pow:
                self.left.computeGrad(outcome,preGradValue*self.right.value*pow(self.left.value,self.right.value-1))
                if self.left.value>0:
                    self.right.computeGrad(outcome,preGradValue*self.value*math.log(self.left.value))
        if self.Type==Type.gradVariable:
            outcome[self.gradVariablePos]+=preGradValue
        

    @classmethod
    def createNum(cls,num):
        outcome=cls()
        outcome.value=num
        return outcome
    
    @classmethod
    def createGradVariable(cls,num,pos):
        outcome=cls()
        outcome.value=num
        outcome.Type=Type.gradVariable
        outcome.gradVariablePos=pos
        return outcome
    
               












