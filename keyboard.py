#!/usr/bin/env python
# coding: utf-8

# # Tones, Frequency and Pitch on a Synthesizer Keyboard
# 
# In this notebook the concepts of **pitch** and **frequency** of a **note** are illustrated on a synthesizer keyboard.  
# For simplicity reasons this notebook synthesizes **pure tones**(sinewaves).   
# We also show the mapping to common **midi** notation.
# 
# Natural sounds from musical instruments or the human voice are much more complex and often (quasi)-periodic signals.  For these sounds the pitch percept corresponds typically to the main periodicity or equivalently to the fundamental frequency.
# For more detailed elaboration look for demos on the subject of *Pitch & Timbre*.
#   
# ##### WARNINGS
# There are some unintended artifacts in the sounds due to the simplistic synthesizing with abrupt start and finish of each tone
# 
# ##### CREDITS
# Elaborated and corrected from [ch11_image/07_synth.ipynb](https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter11_image/07_synth.ipynb)
# in the *IPython Cookbook, 2nd Edition, Copyright (c) 2017 Cyrille Rossant*

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import  Audio, display, clear_output
from ipywidgets import widgets
from functools import partial
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option("precision", 1)
AutoPlay = False


# In[ ]:


# utilities for generating a pure tone (and playing it)
# using a sampling rate of 16kHz
# and a duration of 0msec 
rate = 16000.
duration = .25
t = np.linspace(
    0., duration, int(rate * duration))
def synth(f):
    x = np.sin(f * 2. * np.pi * t)
    return(x)
def play(x,AutoPlay=True,rate=rate):
    display(Audio(x, rate=rate, autoplay=AutoPlay))


# ### Musical Scales
# 
# We are all familiar at least to some extent with notes making up melodies.   But where do these notes come from ?

# #### Synthesize a short sinewave of 440Hz
# This corresponds to A4 in music notation and in MIDI it has a value of 69.

# In[ ]:


play(synth(440),AutoPlay=AutoPlay)


# #### Tones, pitch and musical scales
# Pitch is the more technical term for what you may know as how high a tone sounds like.   
# Expresssing the pitch may be done in a number of different ways.
# - Frequency (in Hz) ... the engineering way
# - Musical Notes  (C, A#, B*b*, ... ) ... the musicians way
#     + these musical notes carry 'relative' information within a single octave
#     + for 'absolute' tonal information we need to add a key, register or octave: A4 ~ 440Hz , A3 ~ 220Hz
# - MIDI numbers ... the synthesizer's way (a semitone scale)   
# 
# Both the musical and MIDI notations used here are based on the Western music scale with 12 semitones in an octave
# 
# A mapping between note (in musical notation), frequency (in Hz) and midi number is shown on the middle (4th) octave of a standard large keyboard with a total of 88 keys 

# In[ ]:


def m2f(m):
    f = 2.**((m-69.)/12.0) * 440.
    return(f)
def f2m(f):
    m=12.*log2(f/440.0)+69.
    return(m)
def note2f(note):
    return(m2f(note2m(note)))
def note2m(note):
    semitones={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
    register=note[-1]
    m = 12 + int(register)*12 + semitones[note[0]]
    if(len(note)>2): 
        if(note[1] =='#'): m+=1
        elif(note[1] =='b'): m-=1
    return(m)


# In[ ]:


# Create a mapping table between different notations for notes in the middle segment
notes = 'C4,C#4,D4,D#4,E4,F4,F#4,G4,G#4,A4,A#4,B4,C5,C#5,D5'.split(',')
midis = [note2m(key) for key in notes]
freqs = [note2f(key) for key in notes]
keyboard = list(zip(notes,freqs,midis))
# pd.DataFrame(keyboard,columns=['Note','Freq(Hz)','MIDI'])


# ### Hit a note on a synthesizer keyboard to hear its pitch

# In[ ]:


get_ipython().run_line_magic('precision', '0')
layout = widgets.Layout(
    width='42px', height='80px',
    border='2px solid black')
layout1 = widgets.Layout(
    width='42px', height='30px',
    border='0px solid blue')
buttons = []
midi_buttons = []
freq_buttons = []
output = widgets.Output()
#output.layout.width='500px'
for note,f,m  in keyboard:
    button = widgets.Button(description=note[0:-1], layout=layout)
    if(note[1]=='#'): button.style.button_color='#BBBBBB'
    else: button.style.button_color='white'
    button.style.font_weight='bold'
    
    midi_button = widgets.Button(description=str(m),layout=layout1)
    freq_button = widgets.Button(description="%.0f"%(f),layout=layout1)

    def on_button_clicked(f, b):
        # When a button is clicked, we play the sound
        # in a dedicated Output widget.
        with output:
            #print("hallo",f)
            clear_output()
            play(synth(f))

    button.on_click(partial(on_button_clicked,f))
    buttons.append(button)
    midi_buttons.append(midi_button)
    freq_buttons.append(freq_button)

# We place all buttons horizontally.
print("The top line shows the MIDI numbers")
print("The second line shows the frequencies of the tones in Hz")
print("The bottom section is a synthesizer keyboard with the notes on it")
print("Hit a note on the keyboard to hear it")
piano = widgets.VBox([widgets.Box(midi_buttons),widgets.Box(freq_buttons),widgets.Box(buttons),output])
piano


# ### Finally play a simple melody

# In[ ]:


melody = ['C4','D4','E4','C4','E4','F4','G4','G4','C4','D4','E4','C4','E4','F4','G4','G4']
y=[]
for note in melody:
      x = synth(note2f(note))
      y = np.append(y,x,axis=0)
play(y)

