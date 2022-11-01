import torch
import torch.nn as nn
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from utils.torch_amp_dummy import autocast

from .bezier_base import BezierBaseNet
from ..builder import MODELS
import math

class attention(nn.Module):
    def __init__(self,in_channels,mid_channels=None):
        super(attention,self).__init__()
        if not mid_channels:
            mid_channels=in_channels
        #self.make_qkv=nn.Conv2d(in_channels,3*in_channels,3,stride=1,padding=1)
        self.make_qkv=nn.Sequential(nn.Conv2d(in_channels,3*in_channels,1,stride=1,padding=0),
                                    nn.Conv2d(3*in_channels,3*in_channels,11,stride=1,padding=5,groups=3*in_channels))

        #self.make_qkv=nn.Conv2d(in_channels,3*in_channels,7,stride=1,padding=)
        #self.make_q=nn.Conv2d(in_channels,in_channels):
        self.head=8
        self.relu=nn.ReLU()
        self.linear=nn.Conv2d(in_channels,in_channels,1,stride=1,padding=0)
        #self.linear=nn.Sequential(nn.Conv2d(in_channels,in_channels,1,stride=1,padding=0),
        #                          nn.Conv2d(in_channels,4*in_channels,1,stride=1,padding=0),
        #                          nn.Conv2d(4*in_channels,in_channels,1,stride=1,padding=0))
        self.temp=int(math.sqrt(in_channels//self.head))
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        # torch.Size([20, 256, 23, 40])
        B,C,H,W=x.shape
        q,k,v=torch.chunk(self.make_qkv(x),3,dim=1)
        q=q.reshape(B,self.head,C//self.head,H,W).reshape(B,self.head,C//self.head,-1).permute(0,1,3,2).contiguous()  #B,h,L,C
        k=k.reshape(B,self.head,C//self.head,H,W).reshape(B,self.head,C//self.head,-1).contiguous() #B,h,C,L
        v=v.reshape(B,self.head,C//self.head,H,W).reshape(B,self.head,C//self.head,-1).permute(0,1,3,2).contiguous() #B,h,L,C

        #B,h,L,c
        attn=self.softmax(torch.matmul(q,k)/self.temp)
        new_x=torch.matmul(attn,v).permute(0,1,3,2).contiguous().reshape(B,C,H*W).reshape(B,C,H,W)
        
        #print("new_x.shape",new_x.shape)
        #print("x.shape",x.shape)
        return self.relu(self.linear(new_x))+x
        #return (0.5*self.linear(new_x))+(0.5*x)


@MODELS.register()
class BezierLaneNet(BezierBaseNet):
    # Curve regression network, similar design as simple object detection (e.g. FCOS)
    def __init__(self,
                 backbone_cfg,
                 reducer_cfg,
                 dilated_blocks_cfg,
                 feature_fusion_cfg,
                 head_cfg,
                 aux_seg_head_cfg,
                 image_height=360,
                 num_regression_parameters=8,
                 thresh=0.5,
                 local_maximum_window_size=9):
        super(BezierLaneNet, self).__init__(thresh, local_maximum_window_size)
        global_stride = 16
        branch_channels = 256

        self.backbone = MODELS.from_dict(backbone_cfg)
        self.reducer = MODELS.from_dict(reducer_cfg)
        self.dilated_blocks = MODELS.from_dict(dilated_blocks_cfg)
        self.simple_flip_2d = MODELS.from_dict(feature_fusion_cfg)  # Name kept for legacy weights
        self.aggregator = nn.AvgPool2d(kernel_size=((image_height - 1) // global_stride + 1, 1), stride=1, padding=0)
        self.regression_head = MODELS.from_dict(head_cfg)  # Name kept for legacy weights
        self.proj_classification = nn.Conv1d(branch_channels, 1, kernel_size=1, bias=True, padding=0)
        self.proj_regression = nn.Conv1d(branch_channels, num_regression_parameters,
                                         kernel_size=1, bias=True, padding=0)
        self.segmentation_head = MODELS.from_dict(aux_seg_head_cfg)
        self.attn1=attention(256)
        self.attn2=attention(256)
        #self.attn3=attention(256)
        print("model is ",self)
        #self.attn2=attention(256)
    def forward(self, x):
        # Return shape: B x Q, B x Q x N x 2
        x = self.backbone(x)
        if isinstance(x, dict):
            x = x['out']
        #print("self.dilated_blocks",self.dilated_blocks)
        #return 
        if self.reducer is not None:
            x = self.reducer(x)
        #add attention
        #print("before x,shape",x.shape)
        x=self.attn1(x)
        x=self.attn2(x)
        #print("after x,shape",x.shape)
        # Segmentation task
        if self.segmentation_head is not None:
            segmentations = self.segmentation_head(x)
        else:
            segmentations = None
        
        #print("here x.shape",x.shape)
        #print("self.dilated_blocks",self.dilated_blocks)
        if self.dilated_blocks is not None:
           x = self.dilated_blocks(x)
        #x=self.attn2(x)
        #x=self.attn3(x)
        with autocast(False):  # TODO: Support fp16 like mmcv
            x = self.simple_flip_2d(x.float())
        #print("x.shape",x.shape)
        #print("x = self.aggregator(x).shape",self.aggregator(x).shape)
        x = self.aggregator(x)[:, :, 0, :]


        #print("here x.shape",x.shape)
        '''
        x.shape torch.Size([20, 256, 23, 40])
        x = self.aggregator(x).shape torch.Size([20, 256, 1, 40])
        here x.shape torch.Size([20, 256, 40])
        '''
        
        x = self.regression_head(x)
        #print("x.shape",x.shape)
        
        
        logits = self.proj_classification(x).squeeze(1)
        #print("logits.shape",logits.shape)
        curves = self.proj_regression(x)
        #print("curves.shape",curves.shape)

        #x.shape torch.Size([20, 256, 40])
        #logits.shape torch.Size([20, 40])
        #curves.shape torch.Size([20, 8, 40])
        #return
        return {'logits': logits,
                'curves': curves.permute(0, 2, 1).reshape(curves.shape[0], -1, curves.shape[-2] // 2, 2).contiguous(),
                'segmentations': segmentations}

    def eval(self, profiling=False):
        super().eval()
        if profiling:
            self.segmentation_head = None
