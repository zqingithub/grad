from gradMatrix import gradMatrix as gm
import torch
import aNNetwork
from aNNetwork import actFunc
from aNNetwork import typeOfLayer as tpLayer
import gradMatrix



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
temp1=gm(torch.tensor([[1,1,1],[1,1,1],[1,1,1]]),grad.Type.gradVariable)
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
#temp1=gm(torch.rand((5,5),dtype=torch.float),grad.Type.gradVariable)
#temp2=gm(torch.rand((2,2),dtype=torch.float),grad.Type.gradVariable)
#temp3=gm(torch.rand((2,3),dtype=torch.float))

temp1=gm(torch.ones(3,3,dtype=torch.float),grad.Type.gradVariable)
#temp2=gm(torch.tensor((6,2),dtype=torch.float),grad.Type.gradVariable)
temp2=gm(torch.ones(3,3,dtype=torch.float),grad.Type.gradVariable)
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
    cnn=aNNetwork.CNN(layer,modelName='leNet-5(CNN)')
    aNNetwork.trainByAdamStep(cnn,trainDS,testDS,10,50000)

main()

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