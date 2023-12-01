'''
Multi-feature Data Generation for DTCO -- A study on WAT and CP
[00] Virtual Silicon Dataset: Utilizing CP and WAT

Copyright (C) 2022, by DipSci Technology Corpopation, LTD. and DigWise Technology, LTD.

This software tool is independently developed and owned by DipSci Technology Co., Ltd. and 
DigWise Technology Co., Ltd. It is freely licensed to industry and academia for use, sharing, 
reproduction, and improvement, on the condition that this copyright notice is retained or 
the original creator is acknowledged in the contributors list. Any derivatives created for 
commercial purposes using this software tool must also include this copyright notice or 
give credit to the original creator in the contributors list.

Revision History:
2021/10/26 hockchen init DTCO
2022/01/15 hockchen init CP-WAT GAN
2022/06/01 hockchen training set preparation
'''
#%% custom data set & utility for analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import torch
from scipy import stats
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.stats import wasserstein_distance

# Virtual Silicon data set and data loader
# NOTE! Rotation and smoothing effects are not satisfactory, temporarily disabled 
class VSDataset(Dataset): # input PIL float(batch,H,W,C), and output training batch uint8(batch,C,H,W) [-1,1]
    def __init__(self, data=None, mask=None, trans=None):
        if data is not None: # normalize all channels to uint8 [0,255]
            nz = np.nan_to_num(data,nan=0,copy=True) # replace nan to zero
            cm = np.nanquantile(nz,[0,1],axis=(0,1,2)).reshape(2,1,1,1,-1) # grab min-max per-channel
            self.data = ((nz-cm[0])/(cm[1]-cm[0]+1e-6)*255).astype(np.uint8) # uint8 [0,255]
            #self.mask = torch.from_numpy(mask.astype(bool)) # convert to tensor for mask operation
            self.transform = transforms.Compose([
                transforms.ToTensor(), # convert uint8(H,W,C) [0,255] to uint8/255 [0,1], otherwise keep the same range
                #transforms.RandomRotation(degrees=(-2, 2),fill=0,interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.Normalize(mean=[0.5], std=[0.5]), # normalize to [-1,1] y = (x-mean)/std
            ]) if trans is None else transforms.Compose(trans)
        self.mask = mask #torch.from_numpy(mask.astype(bool)) # convert to tensor for mask operation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]  # to PIL (H,W,C) uint8 [0,255]
        #x = self.smooth_transform(x,s=(1,3)) # PIL smoothing prior the torch transform (C,H,W)
        x = self.transform(x) # convert numpy x(H,W,C) [0,255] to torch x(C,H,W) [-1,1]
        #x = x.masked_fill(~self.mask, -1) # apply mask (optional)
        return x
    
    # NOTE! will lead to unrealistic data distribution
    def smooth_transform(self,x,s=(1,3)): # x in PIL (H,W,C)
        '''input x should be uint8 [0-255] in PIL(H,W,C) shape, applied prior the torch transform'''
        radius = np.random.randint(s[0],s[1])
        x = uniform_filter(x, size=(radius,radius,1), mode='reflect') # r=0,1 no operation, suggest r<=2
        x = np.where(self.mask.reshape(*self.mask.shape,1), x, 0) # apply mask
        #print(f'random smoothing filter, radius={radius}')
        return x.astype(np.uint8) # to uint8 before applying torch.transform 
    
    def showBatchdata(self,batch,tag='',num=None,ncol=None,dtype='2d',view=(90,270),s=1.8,img=None):
        '''batch (grid data) display of feature surfaces'''
        H,W = batch.shape[1:]
        #num = 25 if batch.shape[0]>25 else batch.shape[0] if num is None else num
        num = num if num is not None else 25 if batch.shape[0]>25 else batch.shape[0]
        ncol = ncol or 10
        nrow = int(np.ceil(num/ncol))
        gx, gy = np.meshgrid(np.linspace(1, W, W), np.linspace(1, H, H))
        plt.figure(figsize=(ncol*s,nrow*s))
        plt.suptitle(f'{tag}',y=0.99,c='b')
        for i in range(num):
            if dtype!='3d':
                ax = plt.subplot(nrow,ncol,i+1,title=f'{i+1}')
                ax.contourf(gx,gy,batch[i],levels=20)
                #x,y,z = gx.ravel(),gy.ravel(),batch[i].ravel()
                #ax.tricontourf(x[z>0],y[z>0],z[z>0],levels=20) # more smooth, compared to ax.contourf
            else:
                ax = plt.subplot(nrow,ncol,i+1,title=f'{i+1}',projection='3d')
                ax.plot_surface(gx,gy,batch[i],antialiased=False,linewidth=0,edgecolors='none',cmap=plt.cm.viridis)
                ax.view_init(*view)
                ax.set_box_aspect((4,4,2), zoom=1.6)
            ax.invert_yaxis() # set origin to top left, 3D view=(90,0)
            ax.axis('off')
        plt.tight_layout(rect=(0,0,1,1))
        if img!=None:
            plt.savefig(img)

    def sigma_percentage(self,sigma):
        '''return left and right % boundary based on the specified sigma value'''
        sta = stats.norm(0,1)
        p = 100-sta.cdf(-1*sigma)*2*100
        h = round((100-p)/2,1)
        h = 0 if h==0 else h
        return h,100-h
    
    def legalizeData(self,x,scale):
        '''input x as PIL(batch,H,W,C)'''
        # normalize generated data [-1,1] to original scale [0,x]
        dmin,dmax = np.zeros_like(scale[0]),scale[1]
        dmin,dmax = dmin.reshape(1,1,1,-1),dmax.reshape(1,1,1,-1)
        xm = np.quantile(x,[0,1],axis=(0,1,2)) # min-max per channel
        x = (x-xm[0])/(xm[1]-xm[0])*(dmax-dmin) + dmin
        x = np.where(self.mask.reshape(1,*self.mask.shape,1),x,np.nan) # apply mask
        return x

    # sigma outlier removal, NOTE! might cause unusuall data distribution
    def filterGridData(self,gd,sigma=2.5,radius=0):
        '''apply sigma outlier removal and average filtering to the griddata (batch,H,W,C) as PIL'''
        #mask = (~np.isnan(gd)).any(axis=(0, 3)) # (H, W)
        print(f'{sigma} sigma outlier removal ...')
        s1,s2 = self.sigma_percentage(sigma)
        m = np.nanpercentile(np.where(gd>0,gd,np.nan),[s1,s2],axis=(1,2)) # (2, B, C)
        m = np.expand_dims(m,axis=(2,3)) # (2, B, 1, 1, C)
        #gt = np.clip(gd, 0, m[1]) # may cause illregular dustribution (local boundary)
        gt = np.where((m[0]<=gd)&(gd<=m[1]), gd, np.nan) # removal might cause data gap
        # smooth filtering +/-radius (Sample Repair for Wafer Defects)
        if radius>0: # kernel size = radius*2+1
            print(f'apply smooth filtering, radius={radius}')
            #gt = np.where(gt>0, gt, 0) # replace nan to 0 for smooth filtering (non-zero might cause illegal samples)
            gt = np.where(gt>0, gt, m[0]) # non-zero might cause illegal samples
            #gt = uniform_filter(gt, size=(1,radius,radius,1), mode='constant', cval=0) # avg.filter on H,W layers
            gt = uniform_filter(gt, size=(1,radius,radius,1), mode='reflect') # avg.filter on H,W layers
        gt = np.where(self.mask.reshape(1,*self.mask.shape,1), gt, np.nan ) # apply mask
        #np.set_printoptions(precision=3, suppress=True)
        print('org vs. new min:\n',np.c_[np.nanmin(np.where(gd>0,gd,np.nan),axis=(0,1,2)),np.nanmin(gt,axis=(0,1,2))].round(3))
        print('org vs. new max:\n',np.c_[np.nanmax(np.where(gd>0,gd,np.nan),axis=(0,1,2)),np.nanmax(gt,axis=(0,1,2))].round(3))
        print('org vs. new samples:\n',np.c_[np.sum(gd>0,axis=(0,1,2)),np.sum(gt>0,axis=(0,1,2))].round(3))
        return gt

    def repairLossData(self,gd,random='clip0',radius=0):
        '''loss data repair for input griddata (batch,H,W,C) as PIL'''
        m = np.nanquantile(np.where(gd>0,gd,np.nan), [0,1],axis=(1,2)) # (2, B, C)
        m = np.expand_dims(m,axis=(2,3)) # (2, B, 1, 1, C)
        if random=='uniform':
            print('fill uniform random')
            x = np.random.uniform(m[0], m[1], size=gd.shape) # uniform random
            x = np.clip(x, m[0], m[1])
            gr = np.where(gd>0, gd, x) # replace nan with random value
        elif random=='normal':
            print('fill normal random')
            x = np.random.normal(m.mean(axis=0),(m[1]-m[0])/10,size=gd.shape) # normal random
            x = np.clip(x, m[0], m[1])
            gr = np.where(gd>0, gd, x) # replace nan with random value
        elif random=='clipm':
            print('clip minimum')
            gr = np.where(gd>0, gd, m[0]) # replace nan with min value
        elif random=='clip0':
            print('clip zero')
            gr = np.where(gd>0, gd, 0) # replace nan with min value
        if radius>0: # smoothing filter
            print(f'smoothing filter: kernel {radius*2+1}*{radius*2+1}')
            gr = np.where(gd>0, gd, m[0]) 
            gr = uniform_filter(gr, size=(1,radius,radius,1), mode='reflect') 
        return np.where(self.mask.reshape(1,*self.mask.shape,1), gr, np.nan) # apply mask
    
    def qualifyScatter(self,gr,gf=None,featureL=None,fx=1,fy=0,sub=100,s=5,alpha=0.2,img=None):
        '''gr := real (or referenced) griddata (B,H,W,C), gf := fake (or target) griddata (B,H,W,C)'''
        featureL = featureL if featureL is not None else [f'F{i}' for i in range(gr.shape[-1])]
        real = np.where(gr>0,gr,np.nan).transpose(3,0,1,2).reshape(gr.shape[-1],-1) # drop nan, mask
        plt.figure(figsize=(7,6))
        plt.suptitle(f'{len(gr[gr>0]):,} real' + ('' if gf is None else f', {len(gf[gf>0]):,} fake'))
        plt.scatter(real[fx,::sub],real[fy,::sub],c='k',s=s,alpha=alpha,label=f'real:{sum(real[0]>0):,}')
        if gf is not None:
            fake = np.where(gf>0,gf,np.nan).transpose(3,0,1,2).reshape(gf.shape[-1],-1) # drop nan, mask
            plt.scatter(fake[fx,::sub],fake[fy,::sub],c='b',s=s,alpha=alpha,marker='x',label='fake:{sum(fake[0]>0):,}')
        plt.xlabel(featureL[fx])
        plt.ylabel(featureL[fy])
        plt.grid(which='major',linestyle='-',zorder=0)
        plt.grid(which='minor',linestyle=':',zorder=0)
        plt.minorticks_on()
        plt.legend(markerscale=1,handles=[
            plt.scatter(np.nan,np.nan,c='k',label='real'),
            plt.scatter(np.nan,np.nan,c='b',label='fake')])
        plt.tight_layout()
        if img!=None:
            plt.savefig(img)
        
    # similarity with Jensen Shannon Divergence & Wasserstein Distance
    def qualifyDistribution(self,gr,gf,featureL,bins=100,img=None):
        '''using JS divergence to quantify similarity'''
        '''gr := real (or referenced) griddata (B,H,W,C), gf := fake (or target) griddata (B,H,W,C)'''
        plt.figure(figsize=(12,6))
        ncol = int((len(featureL)+0.5)//2)
        sL = []
        for i,item in enumerate(featureL):
            vr = gr[:,:,:,i].ravel() # real
            vf = gf[:,:,:,i].ravel() # fake
            vr = vr[vr>0] # drop mask
            vf = vf[vf>0] # drop mask
            binn = np.linspace(min(vr.min(),vf.min()), max(vr.max(),vf.max()), bins) # align bin scale
            hr,x = np.histogram(vr,bins=binn,density=True)
            hf,x = np.histogram(vf,bins=binn,density=True)
            p = hr/np.sum(hr) + 1e-16
            q = hf/np.sum(hf) + 1e-16
            m = 0.5*(p+q) 
            jsdiv = 0.5*(np.sum(p*np.log2(p/m))+np.sum(q*np.log2(q/m))) # kl_divergence
            similarity = 1-jsdiv
            wdist = wasserstein_distance(hr, hf)
            sL += [similarity]
            x = (x[:-1]+x[1:])/2
            ax = plt.subplot(2,ncol,1+i)
            ax.set_title(f'{item}\njs:{similarity:.3f}, wd:{wdist:.3f}',fontsize=10,color='b')
            ax.plot(x, hr, label='real')
            ax.plot(x, hf, label='fake')
            #ax.set_xticks([]) # mask for paper
            ax.set_yticks([]) # mask for paper
            print(f'similarity:{item:10s} = {similarity:.3f}, wasserstein_distance: {wdist:.3f}')
        plt.legend()
        plt.suptitle(f'{(gr.mean(axis=3)>0).sum():,} real, {(gf.mean(axis=3)>0).sum():,} fake (similarity:{np.array(sL).mean():.3f})')
        plt.tight_layout(rect=(0,0,1,1))
        if img!=None:
            plt.savefig(img)
        print('org vs. new samples:\n',np.c_[np.sum(gr>0,axis=(0,1,2)),np.sum(gf>0,axis=(0,1,2))].round(3))
        print(f'average: {np.array(sL).mean():.3f}')

    def qualifyDataConsistency(self,gd,wid=0,ch=0):
        ''''data consistency evaluation between wafer mask and generated griddata (B,H,W,C);
            by default, the first feature (ch=0) of the first wafer (wid=0) will be applied'''
        w = gd[wid,:,:,ch] # wafer data
        t = self.mask # true mask
        H,W = self.mask.shape
        gx,gy = np.meshgrid(range(1,W+1), range(1,H+1))
        plt.figure(figsize=(12,6))
        #plt.suptitle(f'{len(w[w>0]):,} of {len(t[t>0]):,}')
        ax = plt.subplot(121,title=f'Data Consistency (WID:{wid}, ch:{ch})')
        ax.scatter(gx[w>0],gy[w>0],c='r',marker='x',alpha=0.5,label=f'data {len(w[w>0])}')
        ax.scatter(gx[t>0],gy[t>0],facecolor='none',edgecolor='k',marker='s',alpha=0.9,label=f'mask {len(t[t>0])}')
        ax.invert_yaxis() # set origin to top left, 3D view=(90,0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid()
        ax = plt.subplot(122,projection='3d')
        ax.plot_wireframe(gx,gy,w,color='r',alpha=0.3)
        ax.scatter(gx[w>0],gy[w>0],w[w>0],color='r',s=1,alpha=0.5,label='data')
        ax.scatter(gx[t>0],gy[t>0],0,facecolor='none',edgecolor='k',alpha=0.5,label='mask')
        ax.set_box_aspect((4,4,3), zoom=1.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.view_init(90,270)
        ax.invert_yaxis()
        #ax.legend()
        plt.title(f'WID:{wid}')
        plt.tight_layout()

    # projection of multidimensional feature vectors onto probability density
    def featureProjection(self,gd,features,axis=(4,5,0),sub=100,bins=(30,30),view=(20,230)):
        d = gd[:,:,:,axis].T.reshape(3,-1)[:,::sub] # flatten
        d = np.where(d>0,d,np.nan) # drop mask
        x,y,z = d
        idx = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        x,y,z = x[idx],y[idx],z[idx] # drop nan
        H,bx,by = np.histogram2d(x,y,bins=bins,density=True)
        bx,by = (bx[:-1]+bx[1:])/2,(by[:-1]+by[1:])/2
        gx,gy = np.meshgrid(bx,by)
        
        r = 1.5 # extension ratio (projection)
        fig = plt.figure(figsize=(8,9.5))
        ax1 = plt.subplot(211, projection='3d', title=f'{len(x):,} samples')    
        ax1.scatter(x,y,z,s=5,alpha=0.1)
        ax1.scatter(x.max()*r,y,z,s=0.2,c='gray',alpha=0.2) # x scatter projection
        ax1.scatter(x,y.max()*r,z,s=0.2,c='gray',alpha=0.2) # y scatter projection
        ax1.scatter(x,y,0,s=0.2,c='gray',alpha=0.2) # z scatter projection

        wy,wz = np.mgrid[y.min():y.max():10j,z.min():z.max():10j]
        ax1.plot_wireframe(x.max()*r,wy,wz,alpha=0.4) # project surface
        wx,wz = np.mgrid[x.min():x.max():10j,z.min():z.max():10j]
        ax1.plot_wireframe(wx,y.max()*r,wz,alpha=0.4)
        wx,wy = np.mgrid[x.min():x.max():10j,y.min():y.max():10j]
        ax1.plot_wireframe(wx,wy,np.zeros_like(wx),alpha=0.4)
        ax1.set_box_aspect((1,1,0.7), zoom=1.1)
        ax1.set_xlabel(features[axis[0]])
        ax1.set_ylabel(features[axis[1]])
        ax1.set_zlabel(features[axis[2]])
        ax1.view_init(*view)

        ax2 = plt.subplot(212, projection='3d',title='') # density map
        ax2.plot_surface(gx,gy,H.T,cmap=plt.cm.viridis,alpha=0.9)
        ax2.contour(gx,gy,H.T,zdir='x',offset=x.max()*1.1) # x density projection
        ax2.contour(gx,gy,H.T,zdir='y',offset=y.max()*1.1) # y density projection
        ax2.set_box_aspect((1,1,0.3), zoom=1.1)
        ax2.set_zlim(bottom=0)
        ax2.set_xlabel(features[axis[0]])
        ax2.set_ylabel(features[axis[1]])
        ax2.set_zlabel('Density')
        ax2.view_init(*view)
        plt.tight_layout()
        def onRotateEvent(event):
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            fig.canvas.draw()
        #fig.canvas.mpl_connect('motion_notify_event', onRotateEvent)
        fig.canvas.mpl_connect('button_release_event', onRotateEvent)

    def convert2DF(self,data,features,outCSV=None):
        '''convert generated data in PIL style (B,H,W,C) to Dataframe'''
        B,H,W,C = data.shape
        mask = np.any(data>0,axis=(0,3)) # remain H,W axes
        #mask = np.any(data[:,:,:,0]>0,axis=0) # use SIDD layer as MASK
        ix,iy = np.where(mask>0) # index of grid data
        d = data[:,ix,iy,:].reshape(-1,C) # (B,3365,C) remain mask index
        tx, ty = np.tile(ix,B), np.tile(iy,B)
        wid = np.arange(1,B+1).repeat(len(ix)).astype(int)
        df = pd.DataFrame(np.column_stack([wid,tx,ty,d]),columns=['WID','X','Y']+features) 
        print(f'convert generated data to dataframe: {df.shape}')
        if outCSV!=None:
            df.to_csv(outCSV,float_format='%g',index=False)
            print(f'generated data was saved into {outCSV} ...')
        return df #.dropna(subset=['SIDD']) !may contain nan in raw data
