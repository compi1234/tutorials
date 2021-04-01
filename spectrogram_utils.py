import os,sys,io 
import scipy.signal

from urllib.request import urlopen
from IPython.display import display, Audio, HTML
import soundfile as sf
import sounddevice as sd

import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

import librosa

# a few utilities
# routine for reading audio from different inputs
def read_audio_from_url(url):
  fp = io.BytesIO(urlopen(url).read())
  data, samplerate = sf.read(fp,dtype='float32')
  return(data,samplerate)

# by default extract the first channel
def read_mono_from_url(url):
  fp = io.BytesIO(urlopen(url).read())
  data, samplerate = sf.read(fp,dtype='float32')
  data1 = data[:,1].flatten()
  return(data1,samplerate)

# time to index converstions;  inputs can be scalars, lists or numpy arrays  outputs are always numpy arrays
def t2indx(t,samplerate):
  return (np.array(t).astype(float)*float(samplerate)+0.5).astype(int)
def indx2t(i,samplerate):
  return np.array(i).astype(float)/float(samplerate)

#scale=10.0/math.log(10)
DB_EPSILON_KALDI = -69.23689    # scale*math.log(1.19209290e-7)  default flooring applied in KALDI
EPSILON_FLOAT = 1.19209290e-7
    
def spectrogram(y,samplerate=16000,frame_shift=10.,frame_length=30.,preemp=0.97,n_fft=512,window='hamm',output='dB',n_mels=None):
    '''
    spectrogram is a wrapper making use of the librosa() library with some adjustments:
        - frame positioning 
            centered at: k*n_shift + n_shift/2
            #frames:  n_samples // n_shift   , first and last frame partially artificial
        - edge processing (mirroring of input signal) similar to Kaldi / SPRAAK
        - pre-emphasis applied after edge processing

    required arguments:
      y       waveform data (numpy array) 

    optional arguments:
      samplerate   sample rate in Hz, default=16000
      frame_shift  frame shift in msecs, default= 10.0 msecs
      frame_length frame length in msecs, default= 30.0 msecs
      preemp       preemphasis coefficient, default=0.95
      window       window type, default='hamm'
      n_mels       number of mel channels, default=80
      n_fft        number of fft coefficients, default=512
      n_mels         number of mel coefficients, default=None
      output       output scale, default='dB', options['dB','power']
      (amin         flooring applied to power before conversion to dB (default= KALDI EPSILON)  )   

    output:
      spectrogram (in dB)
         
    '''
    n_shift = int(float(samplerate)*frame_shift/1000.0)
    n_length = int(float(samplerate)*frame_length/1000.0)
    if n_fft < n_length :
        print('Warning(Spectrogram): n_fft raised to %d'%n_length)
        n_fft = n_length
    
    # extend the edges by mirroring
    ii = n_shift//2
    n_pad = n_fft//2
    z=np.concatenate((y[0:n_pad][::-1],y,y[:-n_pad-1:-1]))
    z[0]=(1.-preemp)*y[0]
    z[1:]= z[1:] - preemp*z[0:-1]
    y_pre = z[ii:len(z)-ii]
   
    spg_stft = librosa.stft(y_pre,n_fft=n_fft,hop_length=n_shift,win_length=n_length,window=window,center=False)
    spg_power = np.abs(spg_stft)**2
    
    if n_mels == None:   spg = spg_power
    else:                spg = librosa.feature.melspectrogram(S=spg_power,n_mels=n_mels,sr=samplerate) 
        
    if output== 'dB':    return(librosa.power_to_db(spg,amin=EPSILON_FLOAT))
    else:                return(spg)
    
# spectrogram plotting routine with optionally:
#   -- waveform 
#   -- up to 2 segmentations in segmentation panel at the bottom
#   -- optionally a pseudo aligned word transcription in the wav panel
#
def plot_spg(spg=None,wav=None,seg=None,txt=None,figsize=(12,8),spg_scale=2,samplerate=16000,n_shift=160,tlim=None,ShowPlot=True):
    '''plot_spg(): Spectrogram plotting routine
            screen will be built of 3 parts
            TOP:     waveform data (optional) + optional word transcriptions
            MIDDLE:  spectrogram data (at least one required)
            BOTTOM:  segmentations (optional)
    
    Parameters:
        spg         spectrogram (list or singleton) data (required), numpy array [n_param, n_fr] 
        wav         waveform data (optional)
        seg         segmentation (list, singleton or none) plotted in segmentation window at the bottom
                    should be passed as DataFrame, optional
        txt         full segment transcript to be printed in waveform axis
        figsize     figure size (default = (12,8))
        spg_scale   vertical scale of spectrogram wrt wav or seg  (default=2)
        samplerate  sampling rate (default=16000)
        n_shift     frame shift in samples, or equivalently the width of non-overlapping frames
                      this is used for synchronisation between waveform and spectrogram/segmentations
        tlim        segment to render
        ShowPlot    boolean, default=True
                      shows the plot by default, but displaying it can be suppressed for usage in a UI loop
        
     Output:
        fig         figure handle for the plot     


        Notes on alignment:
          The caller of this routine is responsible for the proper alignment between sample stream and frame stream
          (see spectrogram() routine).  By default the full sample stream is plotted.

          spg(n_param,n_fr)    
                  x-range   0 ... nfr-1
                  x-view  [-0.5 , nfr-0.5 ]    extends with +- 0.5
          wavdata(n_samples)
                  x-range   0 ... wavdata
                  x-view    -n_shift/2   nfr*n-shift - n_shift/2   (all converted to timescale)
        '''

    if spg is None:
        print("plot_spg(): You must at least provide a spectrogram")
        return
    if type(spg) is not list: spg = [ spg ]
    nspg = len(spg)
    (n_param,n_fr) = spg[0].shape
    
    if seg is None:
        nseg = 0
    else:
        if type(seg) is not list: seg = [seg]
        nseg = len(seg)
        SegPlot = True       

    WavPlot = False if wav is None   else True
    TxtPlot = False if txt is None   else True
    nwav = 1        if WavPlot       else 0
    
    # make an axes grid for nwav waveform's, nspg spectrogram's, nseg segmentation's
    base_height = 1.0/(nwav+nseg/2.0+nspg*spg_scale)
    nrows = nwav+nspg+nseg
    heights = [base_height]*nrows
    for i in range(0,nspg): heights[nwav+i] = base_height*spg_scale
    for i in range(0,nseg): heights[nwav+nspg+i] = base_height/2.0
    fig = plt.figure(figsize=figsize,clear=True,constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows,ncols=1,height_ratios=heights)
    
    # frame-2-time synchronization on basis of n_fr frames in spectrogram and n_shift
    #    by default it extends the view at the edges by  1/2 nshift samples
    indxlimits = np.array([-n_shift/2, n_fr*n_shift-n_shift/2])
    tlimits = indx2t(indxlimits,samplerate)  

    # add waverform plot
    if WavPlot:
        ax = fig.add_subplot(gs[0,0])
        n_samples = len(wav)
        # if n_samples <= ((n_fr-1) * n_shift):
        #    print("plot_spg() WARNING: waveform too short for spectrogram: %d <= (%d-1) x %d" %
        #          (n_samples, n_fr,n_shift))
        wavtime = np.linspace(0.0, indx2t(n_samples,samplerate), n_samples)
        ax.plot(wavtime,wav)
        wmax = 1.2 * max(abs(wav)+EPSILON_FLOAT)
        ax.set_ylim(-wmax,wmax)
        fshift = indx2t(n_shift,samplerate)
        ax.set_xlim(tlimits)

        ax.tick_params(axis='x',labeltop=True,top=True,labelbottom=False,bottom=False)
        if TxtPlot:
            ax.text(tlimits[1]/2.,0.66*wmax,txt,fontsize=16,horizontalalignment='center')  

    # add spectrograms
    for i in range(0,nspg):
        ax = fig.add_subplot(gs[nwav+i,0])
        ax.imshow(spg[i],cmap='jet',aspect='auto',origin='lower')
        ax.tick_params(axis='x',labelrotation=0.0,labelbottom=False,bottom=True)        
        if (i == nspg-1) & (nseg==0):
            ax.tick_params(axis='x',labelbottom=True)

    # add segmentations
    for i in range(0,nseg):
        ax = fig.add_subplot(gs[nwav+nspg+i,0])
        plot_seg(ax,seg[i],xlim=tlimits,ytxt=0.5,linestyle='dashed',fontsize=10)
        if i != nseg-1:
            ax.tick_params(axis='x',labelbottom=False)

#        plot_seg(ax,seg1,ymin=0.5,ymax=1.0,ytxt=0.75,linestyle='dashed',fontsize=10)
#        plot_seg(ax_seg,seg2,ymin=0.,ymax=0.5,ytxt=0.25,linecolor='r'

    if not ShowPlot: plt.close()
    return(fig)

# routine for plotting the segmentations   
def plot_seg(ax,df,xlim=[0.,1.],ytxt=0.5,linestyle='solid',linecolor='k',fontsize=14,Vlines=True):
    ''' plot_seg(): plots a segmentation to an axis

    ax:   axis
    df:   dataframe with segment data

    xlim:       X-axis range (default: [0 1])
    [ ymin, ymax: Y-axis range (default: [0 1]) ]
    ytxt        height at which to write out the segmentation (default= 0.5)
    Vlines      flag for plotting segmentation lines (default=True)
    linestyle   default='solid'
    linecolor   default='k'
    fontsize    default=14
    ''' 

    # First plot a dummy axis to avoid matplotlib going wild
    ax.imshow(np.zeros((1,1)),aspect='auto',cmap='Greys',vmin=0.,vmax=1) 
    for iseg in range(0,len(df)):
        i1= df['t0'][iseg]
        i2= df['t1'][iseg]
        txt = df['seg'][iseg]
        if(Vlines):
            ax.vlines([i1,i2],0.,1.,linestyles=linestyle,colors=linecolor)
        xtxt = float(i1+(i2-i1)/2.0)
        ax.text(xtxt,ytxt,txt,fontsize=fontsize,horizontalalignment='center')  
        
    ax.tick_params(axis='y',labelleft=False,left=False)
    ax.set_ylim([0.,1.])
    ax.set_xlim(xlim)