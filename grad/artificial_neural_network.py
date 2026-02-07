#实现神经网络的统一架构，用卷积层统一表示全链接层和卷积层，并实现了批量归一化层和循环神经网络
from __future__ import annotations
from ast import Dict, List, Set
import torch
from gradMatrix import gradMatrix as GM
from enum import Enum
import math
import gradMatrix


class actFunc(Enum):
    """
    表示神经网络中每层激活函数的类别
    """
    sigmoid=0
    Non=1
    exp=2
    ReLU=3
    norm=4
    softMax=5
    LReLU=6
    tanh=7
    GELU=8

class lossFunc(Enum):
    """
    表示损失函数的类别
    """
    L1=0
    L2=1
    L_infin=2
    crossEntropy=3

class typeOfLayer(Enum):
    """
    表示神经网络中每一层的类别，每个类别的层都有一个函数负责相应类型层的构造
    input：神经网络的输入层
    conv：卷积层，全连接层也用卷积层表示，全连接层可以看成是通道数为神经元输出个数，高和宽都为1的卷积层
    reshape：变形层，用于改变卷积层的高和宽，通道数会相应改变，通常用于将卷积层变为全连接层
    pool：池化层
    batchNormalization：批量归一化层
    output：输出层
    LSTM：长短时记忆层
    """
    input=0
    conv=1
    reshape=2
    pool=3
    batchNormalization=4
    output=5
    LSTM=6

class poolMethod(Enum):
    """
    池化层使用的池化方法
    maxPool：最值值池化
    avePool：平均值池化
    """
    maxPool=0
    avePool=1

class recurrentLinkMode(Enum):
    """
    表示在循环神经网络中，之前或之后的时序的输入以何种方式贡献当前时序的输出
    add：和当前时序的输入相加
    concatenate：和当前时序的输入直接拼接，通道拼接
    addCon：同时采用相加和拼接两种方式
    """
    add=0
    concatenate=1
    addCon=2

class recurrentTimeWeightModel(Enum):
    """
    表示在循环神经网络中，每个时序的损失值对总的损失函数的权重
    allOne：表示所有时序的权重均相同，且都为1
    average：表示所有的时序的权重均相同，但是所有时序的权重加起来为1
    classifier：用于使用循环神经网络进行分类的场景。在单向循环神经网络中，采用固定变化步长，离最终输出时序越远的时序权重越小，所有时序的权重加起来为1。
                在双向循环神经网络中，则相当于average模式。
    """
    allOne=0
    average=1
    classifier=2

class timeDirection(Enum):
    """
    表示在双向循环神经网络中信息传递的方向。
    passToFuture：信息从过去传向未来
    futureToPass：信息从未来传向过去
    twoDirectionJunction：两个方向的信息进行交汇
    """
    pastToFuture=0
    futureToPass=1
    twoDirectionJunction=2

class layerInfo:
    """
    用于接收神经网络中每一层的类型、通道数、宽和高、激活函数等构造信息，在构造神经网络时使用
    """
    def __init__(self,channelSize:List[int]=None,layerType:typeOfLayer=typeOfLayer.conv,kernelSize:List[List[int]]=None,aFunc:List[actFunc]=None,pMethod:List[poolMethod]=None,\
        channelTable:List[List[int]]=None,shortCircuitAddSet:Set[int]=set(),shortCircuitConcateSet:Set[int]=set(),isRecurrent:bool=False,rcLinkMode:recurrentLinkMode=recurrentLinkMode.concatenate):
        """
        typeOfLayer:该层的类型
        kernelSize:如果是输入层，则表示图片的高和宽，建议输入层采用pytorch官方中对图像的统一表示，即[p，c，h，w]，p：图片的张数，在实现过程中p为1，通过其他方式实现批量处理，c：通道数，h：图片高，w：图片宽。
                   如果是卷积层，则表示卷积核的高和宽，列表中的每一组高和宽代表一个卷积操作，例如[[3,3],[5,5]]表示一共有两组卷积操作，分别使用3×3和5×5的卷积核进行卷积操作。卷积操作后，图像的高和宽不变
                   如是是池化层，则表示池化核的高和宽，这里要求高和宽必须相等，同时该池化层的所有池化操作都必须使用相同尺寸的池化核，即使kernelSize传入了多组池化核大小，例如[[3,3],[5,5]]，在该池化层
                   的多个池化中仍都使用第一个即3×3尺寸的池化核。
                   如果是reshape层，则表示将上一层的图像高和宽改变为给定值，通道数会相应改变，例如上一层为[1,2,4,4],kernelSize设置为[[2,4]]（传入多组只认第一组尺寸）,则表示经过该层后，输出尺寸变为[1,4,2,4]。
        actFunc：该层使用的激活函数，对卷积层、池化层、批量归一化层有效。
        poolMethod：该层使用的池化方法，对池化层有效。
        channelTable：表示该层的每个卷积或池化操作使用上一层的哪些通道作为输入。例如假设该层有2个卷积操作，上一层一共有4个通道，则channelTable=[[0,3],[1,2,3]],表示该层第一个卷积操作使用上一层的第0、3号通道
                      作为输入（通道编号从0开始，第一个通道用0编号），第二个卷积操作使用上一层的第1、2、3号通道作为输入。如果未给定channelTable，则表示使用上一层的所有通道作为输入。池化操作同理。
        shortCircuitAddSet：表示该层和哪些层通过相加的方式进行短接。例如假如shortCircuitAddSet={2,4,6}，则表示该层和从该层开始往上数的第2、4、6层（该层编号为0，1表示该层的上一层，以此类推）进行相加短接。短接后的
                            的结果作为下一层的输入。例如该层和往上数第二层的输出进行相加后，作为往上数第三层的输入。
        shortCircuitAddSet：同shortCircuitAddSet类似，但是短接方式为通道拼接。
        isRecurrent：表示该层是否为循环神经网络中的循环层，即该层的输出，作为下一个时序中对应层的输入（如果神经网络设置为双向循环神经网络，则同时接收下一个和上一个时序的输出当作当前时序对应层的输入）。
        recurrentLinkMode：在循环神经网络中，接收不同时序的输出作为当前时序对应层的输入的方式。

        构造示例：
        输入层：layerInfo(layerType=typeOfLayer.input,channelSize=[2],kernelSize=[[28,28]])，表示输入的图片通道为2，尺寸为28×28。
                layerInfo(layerType=typeOfLayer.input,channelSize=[100],kernelSize=[[1,1]])，表示输入的图片通道为100，尺寸为1×1，可以理解为是把一个100维的向量当作输入特征。
        reshape层：layerInfo(layerType=typeOfLayer.reshape,kernelSize=[[1,1]])，表示把上一层变成一个全连接层，即把图片拉成一个向量。
        卷积层：layerInfo(layerType=typeOfLayer.conv,channelSize=[5,7],kernelSize=[[3,3],[5,5]],aFunc=[actFunc.LReLU,actFunc.sigmoid]),表示该卷积层一共有两个卷积操作，第一个卷积操作的卷积核大小为3×3，输出通道为5个通道，
                激活函数为LReLU函数（如果激活函数未定义则默认使用LReLU函数）。第二个卷积操作的卷积核大小为5×5，输出通道为7个通道，激活函数为sigmoid函数。最终该层会将两个卷积操作得到通道进行拼接，形成最终的5+7=12个通道作为
                该层的输出。
        池化层：layerInfo(layerType=typeOfLayer.pool,pMethod=[poolMethod.avePool,poolMethod.maxPool],aFunc=[actFunc.LReLU,actFunc.sigmoid],kernelSize=[[2,2]])。表示该池化层有两个池化操作，都使用尺寸为2×2的池化核（池化核
                的高和宽必须相等，且所有池化操作的池化核尺寸必须相等），第一个池化操作使用平均池化的池化方法，激活函数为LReLU，第二个池化操作使用最大值池化的池化方法，激活函数为sigmoid。如果不设置pMethod则默认使用平均池化，
                如果不设置aFunc则默认没有激活函数，即只做池化，不做激活。每个池化操作输出的通道数和该池化操作接收的输入通道数相同，接收的输入通道由channelTable参数设置。最终该层会将两个池化操作得到的通道进行拼接，作为该层
                的输出。
        批量归一化层：layerInfo(layerType=typeOfLayer.batchNormalization,aFunc=[actFunc.LReLU]),对上一层进行批量归一化处理，使用LReLU函数作为激活函数。输出的通道数和上一层的通道数相同。如果未设置aFunc，则不使用激活函数，但是
                      仍会设置斜率和偏置（上一层的输出的每个元素对应一组斜率和偏置）模型参数对批量归一化的结果进行调整。
        输出层：layerInfo(layerType=typeOfLayer.output)。每个神经网络都必须设置一个输出层，用于输出最终的模型输出向量。
        LSTM层：layerInfo(layerType=typeOfLayer.LSTM)。使用经典的LSTM结构，如果为双向LSTM循环神经网络，则记忆存储单元的会接收来自过去和未来的双向信息，拼接双向信息后需要对通道进行降维（记忆存储单元的通道数需要和该层输入的通
                道数相同），降维方法为线性映射直接降维，不使用激活函数，线性映射的斜率和偏置作为模型参数学习。
        """
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

def tanh(input:GM)->GM:
    temp=torch.ones(input.value.shape,dtype=torch.float)
    return input.sigmoid()*GM(temp*2)-GM(temp)

def GELU(input:GM)->GM:
    temp=torch.ones(input.value.shape,dtype=torch.float)
    temp2=input.pow(3)
    temp2=temp2*GM(temp*0.044715)
    temp2=input+temp2
    temp2=temp2*GM(temp*math.sqrt(2/math.pi))
    temp2=tanh(temp2)+GM(temp)
    return input*temp2*GM(temp*0.5)

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
    if actFun==actFunc.tanh:
        input=tanh(input)
    if actFun==actFunc.GELU:
        input=GELU(input)
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
    """
    将感知机、卷积神经网络、全连接神经网络、残差神经网络、循环神经网络集成到统一的架构中
    """
    def __init__(self,modelName:str="ANN",batchSize:int=1,numOfMaxRecurrent:int=1,\
        recurrentTimeWeightModel:recurrentTimeWeightModel=recurrentTimeWeightModel.allOne,isBidirectional:bool=False):
        """
        modelName:模型的名称
        batchSize：模型每次训练时接收多少个样本。影响批量归一化层、模型参数梯度的计算。当设置较大时批量归一化层的均值和方差的计算比较具有统计意义，
                   但是某些较复杂的模型在batchSize设置较大时，可能不会收敛到局部最优值。设置较小时，则批量归一化层效果不佳，但复杂模型更易收敛至局
                   部最优解。最终生成的神经网络模型，可以理解为是将一次只接收单个样本的模型复制出batchSize个，然后连接这batchSize个模型的批量归一
                   化层而得到。
        x：用于模型接收样本的特征，一次接收的样本数量由batchSize决定。
        y：用于模型输出预测的样本标签，一次输出的预测样本标签数量由batchSize决定。
        realY：用于模型接收样本的真实标签，一次接收的样本标签数量由batchSize决定。
        lossValue：模型当前时序的损失函数值，是模型计算图最顶端的节点。
        allLossValue：用于记录模型所有时序的损失函数值。
        modelPara：模型的参数列表，训练模型的目的是以损失函数为导向，寻找模型参数的局部最优值。
        modelParaGradStore：在循环神经网络中，计算模型参数的梯度时，用于累加每个时序计算得到的参数梯度。
        layerInfo：用于记录模型每一层的结构信息，即记录模型的结构。
        layerOutput：模型每层的输出，注意每层一共有batchSize个输出，每个输出和其中一个样本对应。
        layerInput：在构造模型时，用于记录预备构造的层的输入，这些输入是进行跨层短接和循环神经网络中链接其他时序的输出后得到的。
        lossFunc：记录模型使用哪种损失函数。
        scDictAdd：用于记录模型中的各层采用相加方式跨层短接的信息，键为当前层的编号（从0开始编号），值为与当前层短接的层的编号组成的列表。
        scDictCon：用于记录模型中的各层采用通道拼接方式跨层短接的信息，键为当前层的编号（从0开始编号），值为与当前层短接的层的编号组成的列表。
        bnMeanValue：模型中各批量归一化层的平均值。
        tempBnMv：在模型构建过程中，用于临时记录批量归一化层的平均值。
        bnDeviation：模型中各批量归一化层的方差。
        tempBnSD：在模型构建过程中，用于临时记录批量归一化层的标准差。
        bnMvEma：用于记录模型在训练过程中，各批量归一化层的平均值的指数移动平均值（在循环神经网络中，不区分不同时序）。在模型训练完后，使用最终的模型预测标签时使用。
                 注意，由于双向循环神经网络在计算过程中，存在从过去到未来和从未来到过去的两个方向的信息传递。因此在储存各时序的平均值时要维护三组数据，一组是过去向
                 未来传递信息时产生的平均值，一组是未来向过去传递信息时产生的平均值，还有一组时综合未来和过去的信息后产生的平均值。
        bnDeEma：用于记录模型在训练过程中，各批量归一化层的方差的指数移动平均值（在循环神经网络中，不区分不同时序）。在模型训练完后，使用最终的模型预测标签时使用。
                 同bnMvEma一样,也要维护三组数据。
        bnMvEmaPerTime：同bnMvEma的作用类似。区别在于在循环神经网络中，每个时序都记录一个对应的指数移动平均值，不同时序对应的值用bnMvEmaPerTime记录。
        bnDeEmaPerTime：同bnDeEma的作用类似。区别在于在循环神经网络中，每个时序都记录一个对应的指数移动平均值，不同时序对应的值用bnDeEmaPerTime记录。
        finishModel：模型训练完后，用于记录最终的模型。主要是将批量归一化层的平均值和方差用常数替代，而不是通过一组批量样本的实时统计量计算得到，使得能够对单样本进行预测。
        preIndexBn：模型的批量归一化层的平均值和方差的指数移动平均值的计算过程中用到的参数，用于设置前一个计算值的权重。
        poolModelPara：模型的池化层的临时模型参数，用于在构建池化层时使用。当有设置池化层的激活函数时，会产生一组斜率和偏置的模型参数，每个池化层的每个池化操作中的每个输入
                       通道都会产生一组参数。
        posOfShortCircuitPara：模型构造时，跨层短接操作中用到的指示模型参数位置的临时变量。
        numOfMaxRecurrent：设置为1，则是非循环神经网络，当大于1时，则神经网络为循环神经网络。具体的数值定义循环神经网络的最大时序。例如：当numOfMaxRecurrent设置为5时，表示
                           循环神经网络最大能够处理5个时序的时间序列数据。
        nowRecurrent：循环神经网络中当前处理样本的时序长度。
        recurrentTimeWeightModel：循环神经网络中各时序的损失函数贡献值的权重模式。
        __recurrentWeight：循环神经网络中各时序的损失函数贡献值的权重，由recurrentTimeWeightModel设置的权重模式决定。
        isBidirectional：设置循环神经网络是否为双向循环神经网络。
        XRecord：用于记录各个时序上的样本特征输入。
        YRecord：用于记录各个时序上的样本预测标签输入。
        RealYRecord：用于记录各个时序上的样本真实标签输入。
        recurrentLeftInput：用于在循环神经网络中接收来自过去时序的信息，同时在计算模型参数梯度时用于向过去时序传递当前及未来时序的梯度。
        recurrentRightInput：用于在循环神经网络中接收来自未来时序的信息，同时在计算模型参数梯度时用于向未来时序传递当前及过去时序的梯度。
        recurrentOutput:循环神经网络中当前时序的循环层的输出（有不同时序的输出作为输入的层）。
        recurrentInputPreGrad：用于在循环神经网络中计算模型参数梯度时，保存其他时序产生的梯度，计算当前时序的模型参数梯度时会用到。
        recurrentInput：用于在构建循环神经网络时，临时存储recurrentLeftInput和recurrentRightInput中的示例，将循环神经网络的过去和未来时序的输入接入同跨层短接的输入接入统一起来。
        rcLeftInputStore：用于储存循环神经网络中，信息从过去向未来传递时，每个时序的循环层产生的输出。
        rcRightInputStore：用于储存循环神经网络中，信息从未来向过去传递时，每个时序的循环层产生的输出。
        rcLeftInputGradStore：用于在计算模型参数梯度的过程中，暂存recurrentLeftInput的梯度，减少一次梯度计算的次数。
        rcRightInputGradStore：与rcLeftInputGradStore类似。
        yClassifier：用于存储循环神经网络输出的分类预测。
        """
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
        self.bnMvEma:List[List[torch.tensor]]=None
        self.bnDeEma:List[List[torch.tensor]]=None
        self.bnMvEmaPerTime:List[List[List[torch.tensor]]]=[None]*numOfMaxRecurrent
        self.bnDeEmaPerTime:List[List[List[torch.tensor]]]=[None]*numOfMaxRecurrent
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
        self.recurrentOutput:List[List[GM]]=[None]*batchSize
        self.recurrentInputPreGrad:List[List[torch.tensor]]=[None]*batchSize
        self.recurrentInput:List[List[GM]]=[None]*batchSize
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

        if layInfo.typeOfLayer==typeOfLayer.LSTM:
            numOfIn=2
            if self.isBidirectional:
                numOfIn=3
            tempShape=[input.shape[1],input.shape[1]*numOfIn+1,1,1]
            self.modelPara.append(GM(torch.zeros(tempShape,dtype=torch.float)/tempShape[1],gradMatrix.variableType.gradVariable))
            self.modelPara.append(GM(torch.rand(tempShape,dtype=torch.float)*0.1/tempShape[1],gradMatrix.variableType.gradVariable))
            self.modelPara.append(GM(torch.zeros(tempShape,dtype=torch.float)/tempShape[1],gradMatrix.variableType.gradVariable))
            self.modelPara.append(GM(torch.zeros(tempShape,dtype=torch.float)/tempShape[1],gradMatrix.variableType.gradVariable))
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
                self.tempBnSD=self.tempBnSD/GM(torch.ones(self.tempBnMV.value.shape,dtype=torch.float)*(self.batchSize-1))
                self.bnDeviation.append(self.tempBnSD)
                self.tempBnSD=(self.tempBnSD+GM(torch.ones(self.tempBnMV.value.shape,dtype=torch.float)*1e-15)).pow(0.5)+GM(torch.ones(self.tempBnMV.value.shape,dtype=torch.float)*1e-15)
                self.__createModelPara(layerNo)
        
            temp=(self.layerInput[batchNo]-self.tempBnMV)/self.tempBnSD
            temp=temp*self.modelPara[-2]+self.modelPara[-1]
            self.layerOutput[batchNo][layerNo]=doActivation(temp,self.layerInfo[layerNo].actFunc[0])
   

    def __createLSTMLayer(self,layerNo:int,batchNo:int):
        if batchNo==0:
            self.__createModelPara(layerNo)
        input:GM=self.layerInput[batchNo]
        self.recurrentLeftInput[batchNo].append(GM(torch.rand(input.value.shape,dtype=torch.float),gradMatrix.variableType.gradVariable))
        self.recurrentRightInput[batchNo].append(GM(torch.rand(input.value.shape,dtype=torch.float),gradMatrix.variableType.gradVariable))
        input=GM(torch.ones((1,1,self.layerInput[batchNo].value.shape[2],self.layerInput[batchNo].value.shape[3]),dtype=torch.float))
        input=GM.cat(self.layerInput[batchNo],input,1)
        input=GM.cat(self.recurrentLeftInput[batchNo][-1],input,1)
        if self.isBidirectional:
            input=GM.cat(input,self.recurrentRightInput[batchNo][-1],1)
        forget:GM=doActivation(input.conv2D(self.modelPara[-1]),actFun=actFunc.sigmoid)
        remember:GM=doActivation(input.conv2D(self.modelPara[-2]),actFun=actFunc.sigmoid)*doActivation(input.conv2D(self.modelPara[-3]),actFun=actFunc.tanh)
        output1:GM=doActivation(input.conv2D(self.modelPara[-4]),actFun=actFunc.sigmoid)
        tempShape=self.layerInput[batchNo].value.shape
        self.recurrentLeftInput[batchNo].append(GM(torch.rand(tempShape,dtype=torch.float),gradMatrix.variableType.gradVariable))
        self.recurrentRightInput[batchNo].append(GM(torch.rand(tempShape,dtype=torch.float),gradMatrix.variableType.gradVariable))
        memory=self.recurrentLeftInput[batchNo][-1]
        if self.isBidirectional:
            memory=memory+self.recurrentRightInput[batchNo][-1]
        tempOutput=memory*forget+remember
        self.layerOutput[batchNo][layerNo]=doActivation(tempOutput,actFun=actFunc.tanh)*output1
        self.recurrentOutput[batchNo].append(self.layerOutput[batchNo][layerNo])
        self.recurrentOutput[batchNo].append(tempOutput)


    def __updateRecurrentOutput(self,layerNo:int,batchNo:int):
        layInfo:layerInfo=self.layerInfo[layerNo]
        if layInfo.isRecurrent:
            self.recurrentOutput[batchNo].append(self.layerOutput[batchNo][layerNo])

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
        if self.layerInfo[layerNo].typeOfLayer==typeOfLayer.LSTM:
            self.__createLSTMLayer(layerNo,batchNo)
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
        """
        isCheckShare设置为True时，会检测构造完后的神经网络的计算图中有没有中间节点没被引用为操作数的情况（理论上计算图中除了最顶端的中间节点外的每个中间节点都应该是计算图中某个
        中间节点的操作数）。但是如果神经网络中包含批量归一化层，则检测过程会比较耗时，此时可以将isCheckShare设置为False跳过检测过程。
        """
        self.modelPara=[]
        self.modelParaGradStore=[]
        self.layerInfo=lyInfo
        self.lossFunc=lsFunc
        for i in range(self.batchSize):
            self.layerOutput[i]=[None]*len(lyInfo)
            self.recurrentLeftInput[i]=[]
            self.recurrentRightInput[i]=[]
            self.recurrentOutput[i]=[]
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
        self.bnMvEma=[[],[],[]]
        self.bnDeEma=[[],[],[]]


        for i in range(len(lyInfo)):
            for j in range(self.batchSize):
                self.__createRecurrent(i,j)
            for j in range(self.batchSize):
                self.__linkShortCircuit(i-1,j)
            for j in range(self.batchSize):
                self.__createLayer(i,j)
            for j in range(self.batchSize):
                self.__updateRecurrentOutput(i,j)
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
            self.bnMvEmaPerTime[i]=[[],[],[]]
            self.bnDeEmaPerTime[i]=[[],[],[]]

        for i in range(len(self.modelPara)):
            self.modelParaGradStore.append(torch.zeros(self.modelPara[i].grad.shape,dtype=torch.float))

        rcInputLen=len(self.recurrentLeftInput[0])
        for i in range(self.numOfMaxRecurrent+1):          
            for j in range(self.batchSize):
                self.rcLeftInputStore[i][j]=[None]*rcInputLen
                self.rcRightInputStore[i][j]=[None]*rcInputLen
                self.rcLeftInputGradStore[i][j]=[None]*rcInputLen
                self.rcRightInputGradStore[i][j]=[None]*rcInputLen

        for i in range(self.batchSize):
            for j in range(len(self.recurrentLeftInput[0])):
                self.recurrentInputPreGrad[i].append(torch.zeros(self.recurrentLeftInput[0][j].grad.shape,dtype=torch.float))

        for i in range(len(self.bnMeanValue)):
            for k in range(3):
                self.bnMvEma[k].append(torch.zeros(self.bnMeanValue[i].value.shape,dtype=torch.float))
                self.bnDeEma[k].append(torch.zeros(self.bnDeviation[i].value.shape,dtype=torch.float))
            for j in range(self.numOfMaxRecurrent):
                for k in range(3):
                    self.bnMvEmaPerTime[j][k].append(torch.zeros(self.bnMeanValue[i].value.shape,dtype=torch.float))
                    self.bnDeEmaPerTime[j][k].append(torch.zeros(self.bnDeviation[i].value.shape,dtype=torch.float))

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
                for k in range(3):
                    self.finishModel.bnMvEma[k][i]=self.bnMvEma[k][i].clone()
                    self.bnMvEma[k][i].zero_()
                    self.finishModel.bnDeEma[k][i]=self.bnDeEma[k][i].clone()
                    self.bnDeEma[k][i].zero_()
                for j in range(self.finishModel.numOfMaxRecurrent):
                    for k in range(3):
                        self.finishModel.bnMvEmaPerTime[j][k][i]=self.bnMvEmaPerTime[j][k][i].clone()
                        self.bnMvEmaPerTime[j][k][i].zero_()
                        self.finishModel.bnDeEmaPerTime[j][k][i]=self.bnDeEmaPerTime[j][k][i].clone()
                        self.bnDeEmaPerTime[j][k][i].zero_()
            for i in range(iter):
                self.setXBatch(trainData,iter*self.batchSize)
                self.computeValue()
            for i in range(len(self.finishModel.bnMeanValue)):
                for k in range(3):
                    temp=self.finishModel.bnMvEma[k][i]
                    self.finishModel.bnMvEma[k][i]=self.bnMvEma[k][i]
                    self.bnMvEma[k][i]=temp
                    temp=self.finishModel.bnDeEma[k][i]
                    self.finishModel.bnDeEma[k][i]=self.bnDeEma[k][i]
                    self.bnDeEma[k][i]=temp
                for j in range(self.finishModel.numOfMaxRecurrent):
                    for k in range(3):
                        temp=self.finishModel.bnMvEmaPerTime[j][k][i]
                        self.finishModel.bnMvEmaPerTime[j][k][i]=self.bnMvEmaPerTime[j][k][i]
                        self.bnMvEmaPerTime[j][k][i]=temp
                        temp=self.finishModel.bnDeEmaPerTime[j][k][i]
                        self.finishModel.bnDeEmaPerTime[j][k][i]=self.bnDeEmaPerTime[j][k][i].clone()
                        self.bnDeEmaPerTime[j][k][i]=temp
        else:
            for i in range(len(self.bnMeanValue)):
                for k in range(3):
                    self.finishModel.bnMvEma[k][i]=self.bnMvEma[k][i].clone()
                    self.finishModel.bnDeEma[k][i]=self.bnDeEma[k][i].clone()
                for j in range(self.finishModel.numOfMaxRecurrent):
                    for k in range(3):
                        self.finishModel.bnMvEmaPerTime[j][k][i]=self.bnMvEmaPerTime[j][k][i].clone()
                        self.finishModel.bnDeEmaPerTime[j][k][i]=self.bnDeEmaPerTime[j][k][i].clone()


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
                    self.rcLeftInputStore[timeNo][i][j]=self.recurrentOutput[i][j].value.clone()
        else:
            for i in range(self.batchSize):
                for j in range(len(self.recurrentRightInput[0])):
                    self.rcRightInputStore[timeNo][i][j]=self.recurrentOutput[i][j].value.clone()

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

    def __updateBnPara(self,timeNo:int,timeDirection:timeDirection):
        for i in range(len(self.bnMeanValue)):
            self.bnMvEma[timeDirection.value][i]=self.preIndexBn*self.bnMvEma[timeDirection.value][i]+(1-self.preIndexBn)*self.bnMeanValue[i].value
            self.bnDeEma[timeDirection.value][i]=self.preIndexBn*self.bnDeEma[timeDirection.value][i]+(1-self.preIndexBn)*self.bnDeviation[i].value
            self.bnMvEmaPerTime[timeNo][timeDirection.value][i]=self.preIndexBn*self.bnMvEmaPerTime[timeNo][timeDirection.value][i]+(1-self.preIndexBn)*self.bnMeanValue[i].value
            self.bnDeEmaPerTime[timeNo][timeDirection.value][i]=self.preIndexBn*self.bnDeEmaPerTime[timeNo][timeDirection.value][i]+(1-self.preIndexBn)*self.bnDeviation[i].value

    def __loadBnPara(self,timeDirection:timeDirection,timeNo:int=0,isUseBnParaPerTime:bool=True):
        if isUseBnParaPerTime:
            for i in range(len(self.bnMeanValue)):
                self.bnMeanValue[i].value=self.bnMvEmaPerTime[timeNo][timeDirection.value][i]
                self.bnDeviation[i].value=self.bnDeEmaPerTime[timeNo][timeDirection.value][i]
        else:
            for i in range(len(self.bnMeanValue)):
                self.bnMeanValue[i].value=self.bnMvEma[timeDirection.value][i]
                self.bnDeviation[i].value=self.bnDeEma[timeDirection.value][i]

    def computeValue(self,isUpdataBnPara:bool=True,isUseBnParePerTime:bool=True):
        self.allLossValue=0.0
        for i in range(self.nowRecurrent):
            self.__loadRecurrentInput(i,isLeftInput=True)
            if self.isBidirectional:
                self.__loadRecurrentInput(i,isLeftInput=False,isLoadZero=True)
            self.__setXBatchPerTime(i)
            self.__setYRealBatchPerTime(i)
            if self.batchSize==1:
                self.__loadBnPara(timeDirection=timeDirection.pastToFuture,timeNo=i,isUseBnParaPerTime=isUseBnParePerTime)
            self.lossValue.computeValue()
            self.allLossValue+=self.lossValue.value[0,0].item()*self.__recurrentWeight[i]
            self.__saveRecurrentInput(i+1,isLeftInput=True)
            self.__saveYBatchPerTime(i)
            if isUpdataBnPara:
                self.__updateBnPara(i,timeDirection.pastToFuture)
        if self.isBidirectional:
            self.allLossValue=0.0
            for i in range(self.nowRecurrent-1,-1,-1):
                self.__loadRecurrentInput(i,isLeftInput=False)
                self.__loadRecurrentInput(i,isLeftInput=True,isLoadZero=True)
                self.__setXBatchPerTime(i)
                self.__setYRealBatchPerTime(i)
                if self.batchSize==1:
                    self.__loadBnPara(timeDirection=timeDirection.futureToPass,timeNo=i,isUseBnParaPerTime=isUseBnParePerTime)
                self.lossValue.computeValue()
                self.__saveRecurrentInput(i-1,isLeftInput=False)
                if isUpdataBnPara:
                    self.__updateBnPara(i,timeDirection.futureToPass)
            for i in range(self.nowRecurrent):
                self.__loadRecurrentInput(i,isLeftInput=True)
                self.__loadRecurrentInput(i,isLeftInput=False)
                self.__setXBatchPerTime(i)
                self.__setYRealBatchPerTime(i)
                if self.batchSize==1:
                    self.__loadBnPara(timeDirection=timeDirection.twoDirectionJunction,timeNo=i,isUseBnParaPerTime=isUseBnParePerTime)
                self.lossValue.computeValue()
                self.allLossValue+=self.lossValue.value[0,0].item()*self.__recurrentWeight[i]
                self.__saveYBatchPerTime(i)
                if isUpdataBnPara:
                    self.__updateBnPara(i,timeDirection.twoDirectionJunction)

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
                self.recurrentOutput[i][j].sumOfGrad=self.recurrentInputPreGrad[i][j].clone()

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