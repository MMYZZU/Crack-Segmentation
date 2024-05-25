import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import paddle
from paddle.nn import functional as F
import random
from paddle.io import Dataset
from visualdl import LogWriter
from paddle.vision.transforms import transforms as T
import warnings
warnings.filterwarnings("ignore")
import time
from time import *
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import paddle.nn as nn


#patch merging
class patchmerging(nn.Layer):
    def __init__(self, ch_in):
        super(patchmerging, self).__init__()
        self.reduction=nn.Linear(ch_in*4,ch_in)
        self.norm=nn.LayerNorm(ch_in*4)
    def forward(self, x):
        B,C,H,W=x.shape
        x0=x[:,:,0::2,0::2]
        #print(1,x0.shape)
        x1=x[:,:,1::2,0::2]
        x2=x[:,:,0::2,1::2]
        x3=x[:,:,1::2,1::2]
        out=paddle.concat(x=[x0,x1,x2,x3],axis=1)
        out1=paddle.reshape(out,[B,C*4,-1])
        out1=paddle.transpose(out1,perm=[0,2,1])
        out2=self.norm(out1)
        out2=self.reduction(out2)
        out2=paddle.transpose(out2,perm=[0,2,1])
        out2=paddle.reshape(out2,[B,C,H//2,W//2])
        return out2




import math
class CA(nn.Layer):
    def __init__(self, ch_in, ch_out,b=1,gama=2):
        super(CA, self).__init__()
        self.ch_in=ch_in
        kernel_size=int(abs((math.log(ch_in,2)+b)/gama))
        if kernel_size % 2:
            kernel_size=kernel_size
        else:
            kernel_size=kernel_size+1
        
        padding=kernel_size // 2

        self.pool=nn.AdaptiveAvgPool2D(output_size=(1,1))

        self.conv=nn.Conv1D(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding)
        self.active=nn.Sigmoid()
    def forward(self, x):
        b,c,h,w=x.shape
        y=self.pool(x)
        y=paddle.reshape(y,[b,c,-1])
        y=paddle.transpose(y,perm=[0,2,1])
        y=self.conv(y)
        y=paddle.transpose(y,perm=[0,2,1])
        y=paddle.reshape(y,[b,c,1,1])
        y=self.active(y)
        y=y*x
        return y

#EFE1
class Conv1(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(Conv1, self).__init__()

        self.conv11=nn.Sequential(
            nn.Conv2D(ch_in,ch_in//2,kernel_size=1,stride=1),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv112=nn.Sequential(
            nn.Conv2D(ch_in*3//2,ch_out,kernel_size=1,stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )
        self.conv113=nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=1,stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

        self.conv31=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=(3,1),stride=1,padding=(1,0)),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv13=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=(1,3),stride=1,padding=(0,1)),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv33=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )
        self.CA=CA(ch_in//2, ch_in//2,b=1,gama=2)
    def forward(self, x):
        x1=self.conv113(x)
        x2=self.conv11(x)
        y=self.conv31(x2)
        y=self.CA(y)
        z=self.conv13(x2)
        z=self.CA(z)
        w=self.conv33(x2)
        out=paddle.concat(x=[y,z,w],axis=1)
        out=self.conv112(out)
        out=out+x1
        return out

#EFE2
class Conv2(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(Conv2, self).__init__()

        self.conv11=nn.Sequential(
            nn.Conv2D(ch_in,ch_in//2,kernel_size=1,stride=1),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv112=nn.Sequential(
            nn.Conv2D(3*ch_in//2,ch_out,kernel_size=1,stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )
        self.conv113=nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=1,stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

        self.conv31=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=(3,1),stride=1,padding=(1,0)),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv13=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=(1,3),stride=1,padding=(0,1)),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv33=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )
        self.CA=CA(ch_in//2, ch_in//2,b=1,gama=2)
    def forward(self, x):
        x1=x
        x2=self.conv11(x)
        y=self.conv31(x2)
        y=self.CA(y)
        z=self.conv13(x2)
        z=self.CA(z)
        w=self.conv33(x2)
        out=paddle.concat(x=[y,z,w],axis=1)
        out=self.conv112(out)
        out=out+x1
        return out


class encoder_block2(nn.Layer):
    def __init__(self, channel_in, channel_out):#channel_in=2channel_out
        super(encoder_block2, self).__init__()
        self.block1 = Conv1(ch_in=channel_in,ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out,ch_out=channel_out)
        #self.pool=nn.MaxPool2D(kernel_size=2,stride=2)
        self.pool=patchmerging(channel_out)
    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.pool(y)
        return y

class encoder_block3(nn.Layer):
    def __init__(self, channel_in, channel_out):#channel_in=2channel_out
        super(encoder_block3, self).__init__()
        self.block1 = Conv1(ch_in=channel_in,ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block4 = Conv2(ch_in=channel_out,ch_out=channel_out)
        #self.pool=nn.MaxPool2D(kernel_size=2,stride=2)
        self.pool=patchmerging(channel_out)
    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.pool(y)
        return y

class encoder_block4(nn.Layer):
    def __init__(self, channel_in, channel_out):#channel_in=2channel_out
        super(encoder_block4, self).__init__()
        self.block1 = Conv1(ch_in=channel_in,ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block4 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block5 = Conv2(ch_in=channel_out,ch_out=channel_out)
        #self.pool=nn.MaxPool2D(kernel_size=2,stride=2)
        self.pool=patchmerging(channel_out)
    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.pool(y)
        return y




#transformer
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2D(inp, oup, kernal_size, stride, 1, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )

class PreNorm(nn.Layer):
    def __init__(self, axis, fn):
        super().__init__()
        self.norm = nn.LayerNorm(axis)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class DWC_conv(nn.Layer):
    def __init__(self,c,h,w):
        super().__init__()
        self.h=h
        self.w=w
        self.conv=nn.Conv2D(c,c,kernel_size=3,stride=1,padding=1,groups=c)
    def forward(self,x):
        B,P,N,C =x.shape
        # ts=paddle.flatten(x,start_axis=1,stop_axis=2)
        # ts=paddle.transpose(ts,perm=[0,1,2])#B C P*N
        ts=paddle.transpose(x,perm=[0,3,1,2])
        ts=paddle.reshape(ts,[B,C,-1])
        ts=paddle.reshape(ts,[B,C,self.h,self.w])
        ts = self.conv(ts)
        ts =paddle.reshape(ts,[B,C,-1])
        ts =paddle.transpose(ts,perm=[0,1,2])
        ts=paddle.reshape(ts,[B,P,N,C])
        return ts


class FeedForward(nn.Layer):#MLP
    def __init__(self, axis, hidden_axis,h,w, dropout=0.):#hidden_axis=mlp_axis
        super().__init__()
        self.h=h
        self.w=w
        self.fc1 =nn.Linear(axis, hidden_axis)
        self.DW=DWC_conv(hidden_axis,self.h,self.w)
        self.norm=nn.LayerNorm(hidden_axis)
        self.act=nn.GELU()
        self.fc2 = nn.Linear(hidden_axis,axis)
    
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.DW(x1)
        x2=x1+x2
        x2 = self.norm(x2)
        x2 = self.act(x2)
        x2 = self.fc2(x2)
        return x2

class Attention(nn.Layer):
    def __init__(self, axis, heads=8, axis_head=64, dropout=0.):
        super().__init__()
        inner_axis = axis_head *  heads
        project_out = not (heads == 1 and axis_head == axis)

        self.heads = heads
        self.scale = axis_head ** -0.5

        self.attend = nn.Softmax(axis = -1)
        self.to_qkv = nn.Linear(axis, inner_axis * 3, bias_attr = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_axis, axis),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
 
        q,k,v = self.to_qkv(x).chunk(3, axis=-1)

        b,p,n,hd = q.shape
        b,p,n,hd = k.shape
        b,p,n,hd = v.shape
        q = q.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))#unfold
        k = k.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))#
        v = v.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))#

        dots = paddle.matmul(q, k.transpose((0, 1, 2, 4, 3))) * self.scale
        attn = self.attend(dots)

        out = (attn.matmul(v)).transpose((0, 1, 3, 2, 4)).reshape((b, p, n,-1))
        return self.to_out(out)

class Transformer(nn.Layer):
    def __init__(self, axis, depth, heads, axis_head, mlp_axis,h,w, dropout=0.):
        super().__init__()
        self.h=h
        self.w=w
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                PreNorm(axis, Attention(axis, heads, axis_head, dropout)),
                PreNorm(axis, FeedForward(axis, mlp_axis,self.h,self.w,dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileViTBlock(nn.Layer):
    def __init__(self, axis, depth, channel, kernel_size, patch_size, mlp_axis,h,w, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.h=h
        self.w=w

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, axis)

        self.transformer = Transformer(axis, depth, 1, 32, mlp_axis,self.h,self.w,dropout)

        self.conv3 = conv_1x1_bn(axis, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)


    def forward(self, x):
        y = x.clone()                      

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # Global representations
        n, c, h, w = x.shape

        x = x.transpose((0,2,3,1)).reshape((n,self.ph * self.pw,-1,c))#unflod
        x = self.transformer(x)
        x = x.reshape((n,h,-1,c)).transpose((0,3,1,2))#flod


        # Fusion
        x = self.conv3(x)
        x = paddle.concat((x, y), 1)
        x = self.conv4(x)
        return x

class Convblock1(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(Convblock1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        ) 
    def forward(self, x):
        y = self.conv(x)
        return y

class transformer(nn.Layer):
    def __init__(self, image_size, axiss,channels, mlp_axis,channel1,h,w, kernel_size=3, patch_size=(2, 2)):
        super(transformer,self).__init__()
        self.h=h
        self.w=w
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        L = 2        
        self.mv1=MobileViTBlock(axiss, 2 , channels, kernel_size, patch_size,mlp_axis,self.h,self.w)
        self.conv1=Convblock1(channel1, channels)
        #self.pool=nn.MaxPool2D(kernel_size=2,stride=2)
        self.pool=patchmerging(channels)

    def forward(self, x):
        x = self.conv1(x)
        y = self.mv1(x)
        y = self.pool(y)
        return y





class conv11_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv11_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv33_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv33_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class AF_block(nn.Layer):
    def __init__(self, ch_in, ch_out,ch_middle):
        super(AF_block, self).__init__()
        self.conv=conv11_block(ch_in*2,ch_in) 
        self.act=nn.Sigmoid()
        self.conv2=conv33_block(ch_in,1)
        #self.conv3=conv33_block(ch_in//2,1)

    def forward(self, a,b):
        c=paddle.concat(x=[a,b],axis=1)
        c=self.conv(c)
        c1 = self.conv2(c)
        #c1 = self.conv3(c1)
        c2=self.act(c1)
        out=c2*a+(1-c2)*b
        return out


 
class CLF_block(nn.Layer):
    def __init__(self, channel_in, channel_out):#channel_in=1024,channel_out=512,channel_middle=channel_out/16
        super(CLF_block, self).__init__()
        self.conv1=nn.Conv2D(in_channels=channel_in,out_channels=channel_out,kernel_size=1,stride=1,padding=0)
        self.conv2=nn.Conv2D(in_channels=channel_out,out_channels=channel_out,kernel_size=1,stride=1,padding=0)
        self.conv3=nn.Conv2D(in_channels=channel_out,out_channels=channel_out,kernel_size=1,stride=1,padding=0)
        self.conv4=nn.Conv2D(in_channels=channel_out,out_channels=channel_out,kernel_size=1,stride=1,padding=0)
        self.act=nn.Softmax()
    def forward(self, a,b):#
        z = paddle.concat(x=[a, b], axis=1)
        z = self.conv1(z)
        B,C,H,W=z.shape
        q=self.conv2(z)
        k=self.conv3(z)
        v=self.conv4(z)
        q=paddle.reshape(q,[B,C,-1])#b c hw
        k=paddle.reshape(k,[B,C,-1])#b c hw
        v=paddle.reshape(v,[B,C,-1])#b c hw
        k=paddle.transpose(k,perm=[0,2,1])#b hw c
        qk=paddle.matmul(q,k)#b c c
        qk=self.act(qk)
        #qk=paddle.transpose(qk,perm=[0,2,1])#b c c
        out=paddle.matmul(qk,v)#b c hw
        out=paddle.reshape(out,[B,C,H,W])
        return out


class MFE_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(MFE_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2D(ch_in*2,ch_in,kernel_size=1,stride=1),
            nn.BatchNorm2D(ch_in),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=1,dilation=1),
            nn.BatchNorm2D(ch_out),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm2D(ch_out),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=4,dilation=4),
            nn.BatchNorm2D(ch_out),
            nn.ReLU()
        )
        self.clf1=CLF_block(channel_in=1024,channel_out=512)
        self.clf2=CLF_block(channel_in=1024,channel_out=512)


    def forward(self,a,b):
        x=paddle.concat(x=[a,b],axis=1)
        x=self.conv(x)
        u=self.conv1(x)
        v=self.conv2(x)
        w=self.conv3(x)
        y = self.clf1(a=u,b=v)
        z = self.clf2(a=y,b=w)
        out=z+x
        return out



class ALFEblock(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(ALFEblock, self).__init__()
        self.conv1=nn.Conv2D(ch_in*2,ch_in,kernel_size=1,stride=1) 
        self.conv2=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.conv3=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.conv4=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.conv5=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.conv6=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.act=nn.Softmax()
        self.pool1=patchmerging(ch_in)
        #self.pool1=nn.MaxPool2D(kernel_size=2,stride=2)
        self.up=nn.UpsamplingBilinear2D(scale_factor=2)


    def forward(self, x): 
        x1=x
        c1=self.pool1(x)
        
        B,C,H,W=c1.shape
        
        q=self.conv2(c1)
        k=self.conv3(c1)
        v=self.conv4(c1)

        q1=paddle.reshape(q,[B,C,-1])#B C HW
        k1=paddle.reshape(k,[B,C,-1])#B C HW
        v=paddle.reshape(v,[B,C,-1])

        q1=paddle.transpose(q1,perm=[0,2,1])#B HW C
        qk1=paddle.matmul(q1,k1)#B HW HW
        qk1=self.act(qk1)
        qk1=paddle.transpose(qk1,perm=[0,2,1])#B HW HW
        out1=paddle.matmul(v,qk1)
        out1=paddle.reshape(out1,[B,C,H,W])

        
        q2=self.conv5(c1)
        k2=self.conv6(c1)
        q2=paddle.reshape(q2,[B,C,-1])
        k2=paddle.reshape(k2,[B,C,-1])
        k2=paddle.transpose(k2,perm=[0,2,1])
        qk2=paddle.matmul(q2,k2)
        qk2=self.act(qk2)
        #qk2=paddle.transpose(qk2,perm=[0,2,1])
        out2=paddle.matmul(qk2,v)
        out2=paddle.reshape(out2,[B,C,H,W])
        
        out=paddle.concat(x=[out1,out2],axis=1)
        out=self.conv1(out)
        #print(1,out.shape)
        out=self.up(out)
        out=out+x1

        return out


class dsconv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(dsconv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1,groups=ch_in),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1,groups=ch_out),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ER_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(ER_block, self).__init__()
        self.conv = dsconv_block(ch_in,ch_out)

    def forward(self, x):
        y = self.conv(x)
        y=y+x
        return y



import paddle.nn as nn
# paddle.set_device('gpu')
# paddle.__version__

class conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.up(x)
        return x



import paddle
import numpy as np
import paddle.nn as nn
from PIL import Image
from paddle.autograd import backward#将Variouable视为backward
import paddle.nn.functional as F
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.initializer import NumpyArrayInitializer

def edge_conv2d(im):
    sobel_kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype='float32')
    sobel_kernel=sobel_kernel.reshape((1,1,3,3))
    sobel_kernel=np.repeat(sobel_kernel, 3, axis=1)
    sobel_kernel=np.repeat(sobel_kernel, 3, axis=0)
    conv_op=nn.Conv2D(3,3,kernel_size=3,padding=1,weight_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel_kernel)))
    edge_dect=paddle.pow(conv_op(fluid.dygraph.to_variable(im)),2)

    sobel_kernel1=np.array([[ 1, 2, 1],[ 0, 0, 0],[-1,-2,-1]],dtype='float32')
    sobel_kernel1=sobel_kernel1.reshape((1,1,3,3))
    sobel_kernel1=np.repeat(sobel_kernel1, 3, axis=1)
    sobel_kernel1=np.repeat(sobel_kernel1, 3, axis=0)
    conv_op1=nn.Conv2D(3,3,kernel_size=3,padding=1,weight_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel_kernel1)))
    edge_dect1=paddle.pow(conv_op1(fluid.dygraph.to_variable(im)),2)

    sobel_kernel2=np.array([[ 2, 1, 0],[ 1, 0,-1],[ 0,-1,-2]],dtype='float32')
    sobel_kernel2=sobel_kernel2.reshape((1,1,3,3))
    sobel_kernel2=np.repeat(sobel_kernel2, 3, axis=1)
    sobel_kernel2=np.repeat(sobel_kernel2, 3, axis=0)
    conv_op2=nn.Conv2D(3,3,kernel_size=3,padding=1,weight_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel_kernel2)))
    edge_dect2=paddle.pow(conv_op2(fluid.dygraph.to_variable(im)),2)

    sobel_kernel3=np.array([[ 0,-1,-2],[ 1, 0,-1],[ 2, 1, 0]],dtype='float32')
    sobel_kernel3=sobel_kernel3.reshape((1,1,3,3))
    sobel_kernel3=np.repeat(sobel_kernel3, 3, axis=1)
    sobel_kernel3=np.repeat(sobel_kernel3, 3, axis=0)
    conv_op3=nn.Conv2D(3,3,kernel_size=3,padding=1,weight_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel_kernel3)))
    edge_dect3=paddle.pow(conv_op3(fluid.dygraph.to_variable(im)),2)
    
    sobel_out = edge_dect+edge_dect1+edge_dect2+edge_dect3
    sobel_out=paddle.sqrt(sobel_out)
    return sobel_out




class Three_Net(nn.Layer):
    def __init__(self, img_ch=3, output_ch=1):
        super(Three_Net, self).__init__()
        
        self.block11=conv_block(ch_in=3,ch_out=64)
    
        self.pool=patchmerging(ch_in=64)
        self.block12=encoder_block2(channel_in=64,channel_out=128)
        self.block13=encoder_block3(channel_in=128,channel_out=256)
        self.block14=encoder_block4(channel_in=256,channel_out=512)

        self.block21 = transformer(image_size=(256, 256), axiss =96, channels=64,channel1=3,patch_size=(8, 8),mlp_axis=96*2,h=256,w=256) 
        self.block22 = transformer(image_size=(128, 128), axiss =192, channels=128,channel1=64,patch_size=(8, 8),mlp_axis=192*2,h=128,w=128)
        self.block23 = transformer(image_size=(64, 64),   axiss =384, channels=256,channel1=128,patch_size=(2, 2),mlp_axis=384*2,h=64,w=64)
        self.block24 = transformer(image_size=(32, 32),   axiss =768, channels=512,channel1=256,patch_size=(2, 2),mlp_axis=768*2,h=32,w=32)

       

        self.AF1=AF_block(ch_in=64,ch_out=64,ch_middle=8)
        self.AF2=AF_block(ch_in=128,ch_out=128,ch_middle=16)
        self.AF3=AF_block(ch_in=256,ch_out=256,ch_middle=32)
        self.AF4=AF_block(ch_in=512,ch_out=512,ch_middle=64)


        self.ALFE64=ALFEblock(ch_in=64,ch_out=64)
        self.ALFE128=ALFEblock(ch_in=128,ch_out=128)
        self.ALFE256=ALFEblock(ch_in=256,ch_out=256)
        self.ALFE512=ALFEblock(ch_in=512,ch_out=512)

        self.ER64=ER_block(ch_in=64,ch_out=64)
        self.ER128=ER_block(ch_in=128,ch_out=128)
        self.ER256=ER_block(ch_in=256,ch_out=256)
        self.ER512=ER_block(ch_in=512,ch_out=512)

        self.Deepest=MFE_block(ch_in=512,ch_out=512)




        
        self.Conv11=conv_block(1024,512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
       

        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
       
        self.Up_conv3 = conv_block(ch_in=256,ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)

        self.Up1 = up_conv(ch_in=64, ch_out=32)

        self.Up_conv2 = conv_block(ch_in=128,ch_out=64)


        self.Conv_1x1 = nn.Conv2D(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        t1 = self.block21(x)#                   64 256 256
        t2 = self.block22(t1)#                  128 128 128
        t3 = self.block23(t2)#                  256 64 64
        t4 = self.block24(t3)#                  512 32 32

        x = edge_conv2d(x)
        x1 = self.block11(x)
        x1 = self.pool(x1)#                      64 256*256

        out1 =self.AF1(a=x1,b=t1)#                            64 256*256
        out1=self.ALFE64(out1)
        out1=self.ER64(out1)

        x2 = self.block12(x1)#                   128 256*256

        out2=self.AF2(a=x2,b=t2)#                              128 128*128
        out2=self.ALFE128(out2)
        out2=self.ER128(out2)
       
   
        x3 = self.block13(x2)#                   256 64*64

        out3=self.AF3(a=x3,b=t3)#                              256 64*64
        out3=self.ALFE256(out3)
        out3=self.ER256(out3)
        
       
        
        x4 = self.block14(x3)#                   512 32*32

        out4 =self.AF4(a=x4,b=t4)#                             512 32*32
        out4=self.ALFE512(out4)
        out4=self.ER512(out4)
        

        #out5=x4+t4
        
        out5=self.Deepest(a=x4,b=t4) 
        d5 = paddle.concat(x=[out5, out4], axis=1)
        d5 = self.Conv11(d5)
        

        d4 = self.Up4(d5)#                     256 64*64
        
        d4 = paddle.concat(x=[d4, out3], axis=1)
        d4 = self.Up_conv4(d4)
        
        
        d3 = self.Up3(d4)#                        128 256*256
       
        d3 = paddle.concat(x=[d3,out2], axis=1)
        d3 = self.Up_conv3(d3)

        
        d2 = self.Up2(d3)#                        64 512*512
       
        d2 = paddle.concat(x=[d2,out1], axis=1)
        d2 = self.Up_conv2(d2)
        

        d2 = self.Up1(d2)
        
        d1 = self.Conv_1x1(d2)#                   2 512*512

        return d1

IMAGE_SIZE = (256, 256)
num_classes = 2
network = Three_Net(img_ch=3, output_ch=num_classes)
model = paddle.Model(network)
model.summary((-1, 3,) + IMAGE_SIZE)
FLOPs = paddle.flops(network, [1, 3, 256, 256], custom_ops= None, print_detail=True)
print(FLOPs)
