import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sklearn.metrics
from PIL import Image
from torchvision import transforms
import skimage
from skimage.color import rgb2hsv
from skimage.morphology import binary_closing
from skimage.transform import resize
import scipy



## MODEL 2 METHODS 

DATES = a = ['29_marzo',
 '14_abril',
 '28_abril',
 '7_mayo',
 '12_mayo',
 '19_mayo',
 '26_mayo',
 '2_junio',
 '11_junio',
 '16_junio',
 '23_junio',
 '2_julio',
 '9_julio',
 '14_julio',
 '23_julio',
 '5_agosto',
 '13_agosto',
 '19_agosto',
 '15_setiembre',
 '24_setiembre',
 '15_octubre',
 '29_octubre',
 '12_noviembre',
 '26_noviembre']

FLIES_DICT = {j:i for i, j in enumerate(DATES)}

def selec_mic(meta_path, idx_meta, cls):
    """
    meta_path: Path of csv
    idx_meta: List of features(array of index)
    cls: Trainnig type ['N', 'P', 'K', 'H']
    """

    df = pd.read_csv(meta_path, index_col=0)
    df['Date']=[i[:-2] for i in df['Date'].values]
    df.Date.replace(FLIES_DICT, inplace=True)
    
    if cls != 'H':
        df = df[(df['Class']==cls+'_Deficiencia') | \
            (df['Class']==cls+'_Control')|\
                (df['Class']==cls+'_Exceso')]
        n2clas={cls+'_Deficiencia':0, cls+'_Control':1, cls+'_Exceso':2}
        df.Class.replace(n2clas, inplace=True)
        
    elif cls == 'H':
        df = df[(df['Class']=='H50%') | \
            (df['Class']=='Control')]
        n2clas={'H50%':0, 'Control':1}
        df.Class.replace(n2clas, inplace=True)
        
    N_FOLD = 5
    if df.shape[0]%5!=0:
        aux_da = int(-(5-df.shape[0]%5))
        print(aux_da, df.iloc[aux_da:,:].shape)
        df = df.append(df.iloc[aux_da:,:])
    print("Classes: ", df.Class.unique())
    idx_meta.append(-1)
    df = df.iloc[:, idx_meta]
    print(f'Datframe with shape of {df.shape}')
    return df


def assertlen(st, n):
    while (len(st) < n):
        st += " "
    return st

# MODEL 1 METHODS AND LOAD

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

class CustomTrain(nn.Module):
    def __init__(self, n_features, n_classes, layers_list, activation=nn.ReLU(), dropout_list=None, batch_norm=True):
        super(CustomTrain, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.layers_list = layers_list
        self.activation = activation
        self.dropout_list = dropout_list
        self.batch_norm = batch_norm
        self.net = []
        self.b_list = []
        if self.dropout_list:
            self.dropout_list = [nn.Dropout(i) for i in self.dropout_list]
        for i in range(len(self.layers_list)):
            if self.batch_norm:
                self.b_list.append(nn.BatchNorm1d(self.layers_list[i]))
            if i==0:
                self.net.append(nn.Linear(self.n_features, self.layers_list[i]))
            else:
                self.net.append(nn.Linear(self.layers_list[i-1], self.layers_list[i]))
        self.last_layer = nn.Linear(self.layers_list[-1], self.n_classes)
        self.net = nn.ModuleList(self.net)
        
    
    def forward(self, x):

        for i, l in enumerate(self.net):
            #print(i)
            x = self.activation(l(x))
            if self.batch_norm:
                x = self.b_list[i](x)
            if self.dropout_list:
                x = (self.dropout_list[i])(x)
            
        x = self.last_layer(x)
        if self.n_classes == 1:
            x = torch.sigmoid(x)
   
        return x
    
class s_view(nn.Module):
    def forward(self,x):
        if len(x.shape)==4:
            self.i_shape=x.shape
            out=x.view(x.shape[0],-1)
        elif len(x.shape)==2:
            out=x.view(self.i_shape)
        return out

class set_conv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pooling=True,batch_norm=True,stride=1):
        super(set_conv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)

        self.comp_layer=nn.ModuleList(
            [nn.Conv2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding)]+\
                [act]+\
                ([nn.MaxPool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else []) +\
                ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])
        )

    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class set_deconv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pooling=True,batch_norm=True,stride=1):
        super(set_deconv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
            self.out_pad=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)
            self.out_pad=1

        self.comp_layer=nn.ModuleList(
            [nn.ConvTranspose2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding,output_padding=self.out_pad)]+\
            [act]+\
            ([nn.MaxUnpool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else []) +\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])
        )
    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class b_encoder_conv(nn.Module):
    def __init__(self,image_channels=3,repr_sizes=[32,64,128,256],
                kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,stride=1):
        super(b_encoder_conv, self).__init__()
        self.repr_sizes=[image_channels]+repr_sizes
        self.stride=[stride for i in range(len(repr_sizes))]
        
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes))]
        else:
            self.kernels=kernel_size
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes))]
        else:
            self.activators=activators
        #pooling
        if isinstance(pooling,bool):
            self.pooling=[pooling for i in range(len(repr_sizes))]
        else:
            self.pooling=pooling
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes))]
        else:
            self.batch_norm=batch_norm
        
        self.im_layers=nn.ModuleList(
            [
                set_conv(repr_in,
                repr_out,
                kernel_size,
                act,
                pooling,
                batch_norm,
                stride
                )
                for repr_in,repr_out,kernel_size,act,pooling,batch_norm,stride in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pooling,
                    self.batch_norm,
                    self.stride
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x
    
class b_decoder_conv(nn.Module):
    def __init__(self,image_channels=3,repr_sizes=[32,64,128,256],
                kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,stride=1):
        super(b_decoder_conv,self).__init__()
        self.repr_sizes=[image_channels]+repr_sizes
        self.repr_sizes=self.repr_sizes[::-1]
        self.stride=[stride for i in range(len(repr_sizes))]
        
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes))]
        else:
            self.kernels=kernel_size
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes))]
        else:
            self.activators=activators
        self.activators=activators[::-1]
        #pooling
        if isinstance(pooling,bool):
            self.pooling=[pooling for i in range(len(repr_sizes))]
        else:
            self.pooling=pooling
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes))]
        else:
            self.batch_norm=batch_norm
        
        self.im_layers=nn.ModuleList(
            [
                set_deconv(repr_in,
                repr_out,
                kernel_size,
                act,
                pooling,
                batch_norm,
                stride
                )
                for repr_in,repr_out,kernel_size,act,pooling,batch_norm,stride in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pooling,
                    self.batch_norm,
                    self.stride
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x
    
#Add batch normalization,dropout
class NN_layer(nn.Module):
    def __init__(self,inp,out,act=nn.ReLU(),batch_norm=True):
        super(NN_layer,self).__init__()
        self.batch_norm=batch_norm
        self.layer=nn.ModuleList(
            [nn.Linear(inp,out)]+([nn.BatchNorm1d(out)] if self.batch_norm else [])+[act]
            )
    def forward(self,x):
        for sl in self.layer:
            x=sl(x)
        return x


class NeuralNet(nn.Module):
    def __init__(self,input_size,output_size,layer_sizes=[300,150,50],
                activators=nn.LeakyReLU(),batch_norm=True):
        super(NeuralNet,self).__init__()
        self.layer_sizes=[input_size]+layer_sizes+[output_size]
        #self.activators=activators

        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(layer_sizes)+1)]
        else:
            self.batch_norm=batch_norm

        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(layer_sizes)+1)]
        else:
            self.activators=activators

        self.layers=nn.ModuleList(
            [
                nn.Sequential(NN_layer(in_size,out_size,act,bat_norm))
                for in_size,out_size,act,bat_norm in zip(
                    self.layer_sizes[:-1],
                    self.layer_sizes[1:],
                    self.activators,
                    self.batch_norm
                )
            ]
        )
    def forward(self,x):
        for l in self.layers:
            x=l(x)
        return x

    

class P_NET(nn.Module):
    def __init__(self,input,w_latent_space_size,z_latent_space_size,y_latent_space_size,layer_sizes,NN_batch_norm=True):
        super(P_NET,self).__init__()
        self.NN_input=input
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.layer_sizes=layer_sizes
        self.NN_batch_norm=NN_batch_norm

        #P(z|w,y)
        self.pz_wy_mu=nn.ModuleList([NeuralNet(self.w_latent_space_size,#W
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes[::-1],
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        ) for i in range(self.y_latent_space_size)])

        self.pz_wy_sig=nn.ModuleList([NeuralNet(self.w_latent_space_size,#W
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes[::-1],
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + relu
                                        batch_norm=self.NN_batch_norm
                                        ) for i in range(self.y_latent_space_size)])
        #P(x|z)
        self.px_z=NeuralNet(self.z_latent_space_size,#Z
                                        self.NN_input,
                                        layer_sizes=self.layer_sizes[::-1],
                                        batch_norm=self.NN_batch_norm
                                        )
    def z_gener(self,w,n_particle=1):
        z_mean=torch.cat([self.pz_wy_mu[i](w).unsqueeze(1) for i in range(self.y_latent_space_size)],dim=1)
        z_logsig=torch.cat([self.pz_wy_sig[i](w).unsqueeze(1) for i in range(self.y_latent_space_size)],dim=1)
        z=self.reparametrization(z_mean,z_logsig,n_particle)
        return z,z_mean,z_logsig

    def x_gener(self,z):
        x=self.px_z(z)
        return x

    def reparametrization(self,mean,logsig,n_particle=1):
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1)).to(self.device)
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1))
        eps=torch.randn_like(mean)
        std=logsig.mul(0.5).exp_()
        sample=mean+eps*std
        return sample
    
    
    
class Q_NET(nn.Module):
    def __init__(self,input,w_latent_space_size,z_latent_space_size,y_latent_space_size,layer_sizes,NN_batch_norm=True):
        super(Q_NET,self).__init__()
        self.NN_input=input
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.layer_sizes=layer_sizes
        self.NN_batch_norm=NN_batch_norm

        #Q(z|x)
        self.qz_x_mu=NeuralNet(self.NN_input,
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        )

        self.qz_x_sig=NeuralNet(self.NN_input,
                                        self.z_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + relu
                                        batch_norm=self.NN_batch_norm
                                        )
        #Q(w|x)
        self.qw_x_mu=NeuralNet(self.NN_input,
                                        self.w_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + identity
                                        batch_norm=self.NN_batch_norm
                                        )

        self.qw_x_sig=NeuralNet(self.NN_input,
                                        self.w_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Identity()],#RELU + relu
                                        batch_norm=self.NN_batch_norm
                                        )
        #P(y|w,z)
        #Input w.shape + z.shape
        #output sigmoid
        # Add small constant to avoid tf.log(0)
        #self.log_py_wz = tf.log(1e-10 + self.py_wz)
        self.py_wz=NeuralNet(self.w_latent_space_size+self.z_latent_space_size,
                                        self.y_latent_space_size,
                                        layer_sizes=self.layer_sizes,
                                        activators=[nn.LeakyReLU() for i in range(len(self.layer_sizes))]+[nn.Softmax(dim=1)],
                                        batch_norm=self.NN_batch_norm
                                        )
    
    def z_infer(self,x,n_particle=1):
        z_mean=self.qz_x_mu(x)
        z_logsig=self.qz_x_sig(x)
        z=self.reparametrization(z_mean,z_logsig,n_particle)
        return z,z_mean,z_logsig

    def w_infer(self,x,n_particle=1):
        w_mean=self.qw_x_mu(x)
        w_logsig=self.qw_x_sig(x)
        w=self.reparametrization(w_mean,w_logsig,n_particle)
        return w,w_mean,w_logsig

    def y_gener(self,w,z,n_particle=1):
        #z,z_mean,z_logsig=self.z_infer(x,n_particle)
        #w,w_mean,w_logsig=self.w_infer(x,n_particle)
        py=self.py_wz(torch.cat((w,z),dim=1))
        return py

    def reparametrization(self,mean,logsig,n_particle=1):
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1)).to(self.device)
        #eps=torch.randn_like(mean.expand(n_particle,-1,-1))
        eps=torch.randn_like(mean)
        std=logsig.mul(0.5).exp_()
        sample=mean+eps*std
        return sample

class GMVAE(nn.Module):
    def __init__(self,
                 image_dim=int(4096/2),
                 image_channels=3,
                 repr_sizes=[32,64,128,256],
                 layer_sizes=[300],
                 w_latent_space_size=20,
                 z_latent_space_size=20,
                 y_latent_space_size=20,
                 conv_kernel_size=5,
                 activators=[nn.Tanh(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
                 conv_pooling=True,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=1,
                 device="cpu",
                 Multi_GPU=False,
                 in_device="cpu"
                ):
        super(GMVAE,self).__init__()

        self.parallelized=Multi_GPU
        self.in_device=in_device
        self.losses={}

        self.conv_pooling=conv_pooling
        self.conv_batch_norm=conv_batch_norm
        self.NN_batch_norm=NN_batch_norm
        self.conv_kernel_size=conv_kernel_size
        self.activators=activators

        self.layer_sizes=layer_sizes
        self.NN_input=(self.compute_odim(image_dim,repr_sizes,stride=stride)[0]*self.compute_odim(image_dim,repr_sizes,stride=stride)[1])*repr_sizes[-1]
        self.w_latent_space_size=w_latent_space_size
        self.z_latent_space_size=z_latent_space_size
        self.y_latent_space_size=y_latent_space_size
        self.device=device
        
        self.encoder_conv=b_encoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )

        self.P=P_NET(input=self.NN_input,
                    w_latent_space_size=self.w_latent_space_size,
                    z_latent_space_size=self.z_latent_space_size,
                    y_latent_space_size=self.y_latent_space_size,
                    layer_sizes=self.layer_sizes,
                    NN_batch_norm=self.NN_batch_norm
                    )

        self.flatten=s_view()

        self.Q=Q_NET(input=self.NN_input,
                    w_latent_space_size=self.w_latent_space_size,
                    z_latent_space_size=self.z_latent_space_size,
                    y_latent_space_size=self.y_latent_space_size,
                    layer_sizes=self.layer_sizes,
                    NN_batch_norm=self.NN_batch_norm
                    )

        self.decoder_conv=b_decoder_conv(image_channels=image_channels,
                                        repr_sizes=repr_sizes,
                                        kernel_size=self.conv_kernel_size,
                                        activators=self.activators,
                                        pooling=self.conv_pooling,
                                        batch_norm=self.conv_batch_norm,
                                        stride=stride
                                        )
        
        if self.parallelized:
            self.encoder_conv.to('cuda:0')
            self.Q.to('cuda:1')
            self.flatten.to('cpu')
            self.P.to('cuda:2')
            self.decoder_conv.to('cuda:3')
        
    def compute_odim(self,idim,repr_sizes,stride):
        if isinstance(self.conv_pooling,bool):
            pool_l=[self.conv_pooling for i in range(len(repr_sizes))]
        else:
            pool_l=self.conv_pooling

        odim=idim
        for i in range(len(repr_sizes)+np.sum(np.array(pool_l).astype(int))):
            if stride==1:
                odim=conv_output_shape(odim,kernel_size=self.conv_kernel_size, stride=stride, pad=0, dilation=1)
            elif stride==2:
                odim=conv_output_shape(odim,kernel_size=self.conv_kernel_size, stride=stride, pad=int((self.conv_kernel_size-1)/2), dilation=1)
        return odim

    def reparametrization(self,mu,logvar):
        std=logvar.mul(0.5).exp_()
        
        esp=torch.randn(*mu.size()).to(self.device)
        z=mu+std*esp
        return z
        
    
    def forward(self,x):
        #ENCODER
        x=self.encoder_conv(x)
        x=self.flatten(x)
        #FCNN
        #Q(z|x)
        qz_x_mu=self.Q.qz_x_mu(x)
        qz_x_logsig=self.Q.qz_x_sig(x)
        qz=self.Q.reparametrization(qz_x_mu,qz_x_logsig)
        #Q(w|x)
        qw_x_mu=self.Q.qw_x_mu(x)
        qw_x_logsig=self.Q.qw_x_sig(x)
        qw=self.Q.reparametrization(qw_x_mu,qw_x_logsig)
        #P(y|w,z)
        print("debug")
        print(qw.shape)
        print(qz.shape)
        print(torch.cat((qw,qz),dim=1).shape)
        py=self.Q.py_wz_sig(torch.cat((qw,qz),dim=1))

        
        #z=self.reparametrization(mu,sig)
        #DECODER
        #P(z|w,y)
        #opcional
        #pz_mu=self.P.pz_wy_mu(qw)
        #pz_logsig=self.P.pz_wy_sig(qw)
        #P(x|z)
        x_recon=self.P.px_z(qz)
        x_recon=self.reparametrization(x_recon,qw_x_logsig)

        x_recon=self.flatten(x_recon)
        x_recon=self.decoder_conv(x_recon)
        
        return x_recon,qw_x_mu,qw_x_logsig,qz_x_mu,qz_x_logsig,py

    def reconstruction_loss(self,r_x,x):
        BCE=F.mse_loss(r_x,x,reduction='mean')
        return BCE

    def conditional_prior(self,z_x,z_x_mean,z_x_logvar,y_wz,z_wy,z_wy_mean,z_wy_logvar):
        #TODO: self.particles

        z_x_var=z_x_logvar.mul(0.5).exp_() #[batch,z_dim]
        logq=-0.5*torch.mean(z_x_logvar)-0.5*torch.mean((z_x-z_x_mean)**2/(z_x_var**2))
        
        
        z_wy_var=z_wy_logvar.mul(0.5).exp_() #[batch,K,z_dim]
        log_det_sig=torch.mean(z_wy_logvar,dim=2) #[batch,K]
        MSE=torch.mean((z_wy-z_wy_mean)**2/(z_wy_var**2),dim=2) #[batch,K]
        logp=-0.5*log_det_sig-0.5*MSE #[batch,K]
        yplogp=torch.mean(logp.mul(y_wz)) #[batch,K]
        #cond_prior=logq-yplogp
        cond_prior=torch.abs(logq-yplogp)
        return cond_prior

    def w_prior(self,w_x_mean,w_x_logvar):
        w_x_var=w_x_logvar.mul(0.5).exp_() #[batch,z_dim]
        KL_w=0.5*torch.mean(w_x_var**2+w_x_mean**2-1-w_x_logvar)
        return KL_w

    def y_prior(self,py):
        y_prior=torch.mean(torch.sum(py * ( np.log(self.y_latent_space_size,dtype="float32") + torch.log(py) )))
        return y_prior

    def forward_recc_d(self,x_i):
        x=self.encoder_conv(x_i)
        x=self.flatten(x)
        #inference
        z_x,_,_=self.Q.z_infer(x)
        #_=self.Q.y_gener(w_x,z_x) #[batch,K]
        #Generation
        #_,_,_=self.P.z_gener(w_x) #[batch,K,z_dim]
        x_mean=self.P.x_gener(z_x)
        #CNN decoding
        x_mean=self.flatten(x_mean)
        x_mean=self.decoder_conv(x_mean)
        return x_mean

    def forward_recc_u(self,x_i):
        x=self.encoder_conv(x_i)
        x=self.flatten(x)
        #inference
        z_x,_,_=self.Q.z_infer(x)
        w_x,_,_=self.Q.w_infer(x)
        py_wz=self.Q.y_gener(w_x,z_x) #[batch,K]
        #Generation
        z_wy,_,_=self.P.z_gener(w_x) #[batch,K,z_dim]
        x_mean=self.P.x_gener(z_wy[:,torch.argmax(py_wz)])
        #CNN decoding
        x_mean=self.flatten(x_mean)
        x_mean=self.decoder_conv(x_mean)
        return x_mean
    
    def ELBO(self,x_i):
        #CNN encoding
        x=self.encoder_conv((x_i.to(self.in_device if self.parallelized else x_i)))
        x=self.flatten((x.to('cpu') if self.parallelized else x))

        #inference
        z_x,z_x_mean,z_x_logvar=self.Q.z_infer((x.to('cuda:1') if self.parallelized else x))
        w_x,w_x_mean,w_x_logvar=self.Q.w_infer((x.to('cuda:1') if self.parallelized else x))
        py_wz=self.Q.y_gener(
                            (w_x.to('cuda:1') if self.parallelized else w_x),
                            (z_x.to('cuda:1') if self.parallelized else z_x)
                            ) #[batch,K]
        #Generation
        z_wy,z_wy_mean,z_wy_logvar=self.P.z_gener((w_x.to('cuda:2') if self.parallelized else w_x)) #[batch,K,z_dim]
        x_mean=self.P.x_gener((z_x.to('cuda:2') if self.parallelized else z_x))

        #CNN decoding
        x_mean=self.flatten((x_mean.to('cpu') if self.parallelized else x_mean))
        x_mean=self.decoder_conv((x_mean.to('cuda:3') if self.parallelized else x_mean))

        reconstruction=self.reconstruction_loss(
            (x_mean.to("cuda:0") if self.parallelized else x_mean),
            (x_i.to("cuda:0") if self.parallelized else x_i)
            )
        conditional_prior=self.conditional_prior(
            (z_x.to("cuda:0") if self.parallelized else z_x),
            (z_x_mean.to("cuda:0") if self.parallelized else z_x_mean),
            (z_x_logvar.to("cuda:0") if self.parallelized else z_x_logvar),
            (py_wz.to("cuda:0") if self.parallelized else py_wz),
            (z_wy.to("cuda:0") if self.parallelized else z_wy),
            (z_wy_mean.to("cuda:0") if self.parallelized else z_wy_mean),
            (z_wy_logvar.to("cuda:0") if self.parallelized else z_wy_logvar)
            )
        w_prior=self.w_prior(
            (w_x_mean.to("cuda:0") if self.parallelized else w_x_mean),
            (w_x_logvar.to("cuda:0") if self.parallelized else w_x_logvar)
            )
        y_prior=self.y_prior((py_wz.to("cuda:0") if self.parallelized else py_wz))
        loss=reconstruction\
            +conditional_prior\
            +w_prior\
            +y_prior
        #BUILD LOSSES DICT
        self.losses['conditional_prior']=conditional_prior
        self.losses['w_prior']=w_prior
        self.losses['y_prior']=y_prior
        self.losses['reconstruction']=reconstruction
        self.losses["total_loss"]=loss
        
        return self.losses


 ################################################################MODEL############################################


######################################################################################

def seg_mask(img,squared=False):
    hsv_i=rgb2hsv(img)
    h=hsv_i[:,:,0]
    h_b=np.logical_and(h>0.2,h<0.4)
    
    #Erosion and dilation
    disk=skimage.morphology.disk(5)
    hl=np.ones((1,100)).astype('uint8')
    cimg=skimage.morphology.erosion(h_b,disk)
    cimg=binary_closing(cimg,hl)
    cimg=binary_closing(cimg,hl.T)

    limg=skimage.measure.label(cimg)
    props=skimage.measure.regionprops(limg)

    cimg=((limg==np.argmax(np.vectorize(lambda p:p.area)(np.array(props)))+1)).astype("int")

    #Fill
    mask=scipy.ndimage.binary_fill_holes(cimg).astype("int")
    
    x_min=np.min(np.where(mask==1)[0])
    y_min=np.min(np.where(mask==1)[1])
    x_max=np.max(np.where(mask==1)[0])
    y_max=np.max(np.where(mask==1)[1])
    
    #if squared:
    smk=np.zeros(mask.shape)
    Lx=x_max-x_min
    Ly=y_max-y_min
    if Lx<smk.shape[1] and Ly<smk.shape[0]:
        if (Lx)>(Ly):
            d=np.floor((abs((Ly)-(Lx)))/2)
            dL=d
            dR=d+2*((abs((Ly)-(Lx)))/2-d)
            if (y_min-dL)<=0:
                y_max=int(y_max+dR-(y_min-dL))
                y_min=0
            elif (y_max+dR)>=smk.shape[1]:
                y_min=int(y_min-(dL+((y_max+dR)-smk.shape[1])))
                y_max=smk.shape[1]
            else:
                y_min=int(y_min-dL)
                y_max=int(y_max+dR)
        else:
            d=np.floor((abs((Ly)-(Lx))/2))
            dL=d
            dR=d+2*((abs((Ly)-(Lx)))/2-d)
            if (x_min-dL)<=0:
                x_max=int(x_max+dR-(x_min-dL))
                x_min=0
            elif (x_max+dR)>=smk.shape[0]:
                x_min=int(x_min-(dL+((x_max+dR)-smk.shape[0])))
                x_max=smk.shape[0]
            else:
                x_min=int(x_min-dL)
                x_max=int(x_max+dR)
    
    if squared:
        smk[x_min:x_max,y_min:y_max]=1
        mask=smk.astype("int")
        
    return mask,x_max,x_min,y_max,y_min


def trans_emul(img):
    t = transforms.ToTensor()
    image = np.array(Image.open(img))
    mk,x_max,x_min,y_max,y_min=seg_mask(image,squared=False)
    simg=(np.stack((mk,mk,mk),axis=2))*image
    image = simg[x_min:x_max,y_min:y_max,:]
    image = resize(image,(512,512))
    image = rgb2hsv(image)[:,:,0]
    image = (t(image)).to(torch.float)
    
    return image

def img_to_meta(img, model):
    model.eval()
    x_ = model.encoder_conv(img)
    x = model.flatten(x_)
    qz_x_mu=model.Q.qz_x_mu(x)
    qz_x_logsig=model.Q.qz_x_sig(x)
    qz=model.Q.reparametrization(qz_x_mu,qz_x_logsig)
    qw_x_mu=model.Q.qw_x_mu(x)
    qw_x_logsig=model.Q.qw_x_sig(x)
    qw=model.Q.reparametrization(qw_x_mu,qw_x_logsig)
    py=model.Q.py_wz(torch.cat((qw,qz),dim=1))
    return torch.cat((qz_x_mu.flatten(), qz_x_logsig.flatten(), qw_x_mu.flatten(), qw_x_logsig.flatten(), py.flatten()), dim=0)


################################################################################## TEST

#i_test = trans_emul("C:\\users\\abdig\\Downloads\\23_junio_1__arbol_3_fila_A.JPG")

def predict_status(img_path):
    #print("Processing")
    img = trans_emul(img_path)
    #print("Transform done")
    model1=GMVAE(image_dim=int(512),
        image_channels=1,
        repr_sizes=[3,6,12,24,48],
        layer_sizes=[200,100,50],
        w_latent_space_size=10,
        z_latent_space_size=10,
        y_latent_space_size=12,
        conv_kernel_size=7,
        conv_pooling=False,
        activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
        conv_batch_norm=True,
        NN_batch_norm=True,
        stride=2,
        device="cpu")

    model1.load_state_dict(torch.load("./ml_models/best0.pt", map_location=torch.device('cpu')))
    print("Model Loaded")
    out1 = img_to_meta(torch.unsqueeze(img, dim=0), model1)
    print("forward 1")
    
    #Nmodel = CustomTrain(0,0,[0])
    #Kmodel = CustomTrain(0,0,[0])
    #Pmodel = CustomTrain(0,0,[0])
    #Hmodel = CustomTrain(0,0,[0])

    Nmodel = torch.load('./ml_models/Nmodel.pth')
    Kmodel = torch.load('./ml_models/Kmodel.pth')
    Pmodel = torch.load('./ml_models/Pmodel.pth')
    Hmodel = torch.load('./ml_models/Hmodel.pth')
    #print("Model 2 Loaded")
    Nidx = [2, 0, 8, 7, 19, 23, 6, 22, 24, 3, 9, 39, 32, 47, 11, 43, 13, 1, 21, 38, 15, 49, 50, 30, 42, 14, 31, 26, 20, 35, 29, 37, 51, 46, 18, 16, 25, 5, 40, 12, 48, 28, 27, 34, 44, 17]
    Kidx = [2, 0, 6, 8, 13, 1, 4, 44, 7, 30, 10, 48, 19, 28, 18, 21, 29, 27, 51, 32, 17, 50, 46, 43, 9, 35, 26, 36, 12, 15, 34, 37, 11, 41, 20]
    Pidx = [2, 8, 0, 7, 4, 19, 9, 10, 12, 17]
    Hidx = [14, 9, 16, 2, 8, 5, 1, 34, 15, 0, 31, 7, 11, 51, 49, 40, 18, 30, 3, 41]

    Nout = Nmodel(out1[Nidx])
    Kout = Kmodel(out1[Kidx])
    Pout = Pmodel(out1[Pidx])
    Hout = Hmodel(out1[Hidx])
    #print("Forward 2")
    m = nn.Softmax(dim = 0)
    Nout = m(Nout)
    Kout = m(Kout)
    Pout = m(Pout)
    Hout = Hout.item()
    #Hout = m(Hout)
    #print(Nout, Kout, Pout, Hout)
    Nout = Nout.detach()
    Kout = Kout.detach()
    Pout = Pout.detach()
    Ni = torch.argmax(Nout)
    Ki = torch.argmax(Kout)
    Pi = torch.argmax(Pout)
    fin_dict = {
        Ni:Nout[Ni],
        Ki:Kout[Ki],
        Pi:Pout[Pi],
        int(Hout>0.5): Hout
    }
    #print("End")
    #print(fin_dict)
    return fin_dict




