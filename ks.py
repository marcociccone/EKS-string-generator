"""
Created on 07/08/2014

@author: Marco Ciccone
"""

from random import random
from scipy.io.wavfile import write
from pylab import *
from operator import add
from scipy.signal import lfilter, freqz
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

VERBOSE = True


def lagrange(N, delay):
    h = [1 for i in range(N+1)]
    for n in range(N+1):
        for k in range(N+1):
            if k != n: 
                h[n] = h[n] * (delay-k)/(n-k)
    return h

"""
# This is the core of the algorithm
# The delay line is updated averaging the current sample and the  previous one
# 0.996 is the damping factor
"""
def karplus_strong_diff_eq(f0,Fs,duration):
    
    period = int(round(Fs/f0))
    period_frac = Fs/f0
    N = Fs*duration
    
    # original karplus-strong with S = 0.5 , weighted ks with others values
    rho = 0.996
    #rho = 1.0
    S = 0.5
    b0 = 1-S
    b1 = S
    
    
    t_60 = 4 
    rho = 0.001**(1.0/(f0*t_60))
    B = 0 # brightness factor
    
    # one-zero damping filter
    #b1 = 0.5*B # S = B/2
    #b0 = 1-b1 # 1-S
    
    # two-zero damping filter
    #b0 = (1.0 + B)/2
    #b1 = (1.0 - B)/4

    g0 = rho*b0
    g1 = rho*b1


    # pluck (excitation) is made by filling the delay line with random values  
    t0 = time.clock()
    delayline = [random() - 0.5 for i in range(period)]
    
    samples = []
    
    
    N_lagrange = 4 
    delay = 0#Fs/f0 - int(Fs/f0) + N_lagrange/2 - 0.5
    print("delay",delay) 
    h_lagrange = lagrange(N_lagrange, delay )
    coeff = np.convolve([g0,g1], h_lagrange)
    print(coeff)
    print([g0,g1])
    
    for n in range(N) :
        samples.append(delayline[0])
        
        # one-zero damping filter and original KS (INTERPOLATED)
        
        j=0
        interpolated = 0
        for c in coeff:
            interpolated += c*delayline[j] 
            j+=1 
        delayline.append(interpolated)
        
        # one-zero damping filter and original KS (NOT INTERPOLATED)
        #delayline.append( g0 * delayline[0] + g1 * delayline[1])
        
        # two-zero damping filter
        #delayline.append( g1 * delayline[0] + g0 * delayline[1] + g1 * delayline[2])
        delayline.pop(0)
        
    print ("Elapsed time: ", time.clock() - t0, "s")
    
    return samples,delayline

""" 
Initialization of the delay line with samples of white noise
random samples have the physical meaning of string displacement
"""
def excitation(period) :
    delayline = np.array([random() - 0.5 for _ in range(period)])
    return delayline
    
"""
Definition of the low pass filter, unity-dc-gain one pole : up-pick , down-pick
:p takes two different values 0 or 0.9 depending on the pick direction
"""
def pick_direction(delayline,p=0.9) :
    if VERBOSE:
        print("pick_direction : "," angle = ",p)
        
    b_pickdir = [1-p]
    a_pickdir = [1,-p]
    return lfilter(b_pickdir,a_pickdir,delayline)

"""
Definition of the comb filter for the position of the pick
beta is the position of the pick: 0 is the "bridge", 1 is the "nut"
"""
def pick_position(delayline,D,beta=0.5) :
    if VERBOSE:
        print("pick_position : "," pickpos = ",beta)
        
    P = D
    pickpos = int(math.floor(P*beta)) # Pick position
    b_pickpos = np.array([1.0] + ([0]*(pickpos-1)) + [-1]) # Numerator zeros coefficients
    a_pickpos = np.array([1.0]) # denominator poles coefficients
    return lfilter(b_pickpos,a_pickpos,delayline)

"""
Dynamic level low pass filter discretized with the bilinear transform
The spectral centroid tipically rises as plucking/striking becomes more energetic
"""
def dynamic_level_lpfilter(f0,Fs,samples,L) :
    if VERBOSE:
        print("dynamic_level_lpfilter : "," L = ",L)
        
    # w1 = 2*pi*f0 # fundamental frequency in radians per second
    # w1_tilde = w1/(Fs*2)
    # Contracted formula
    w1_tilde = math.pi*f0/Fs
    
    b_dyn = np.array([w1_tilde, w1_tilde])
    a_dyn = np.array([(1+w1_tilde),w1_tilde-1])
    y = lfilter(b_dyn,a_dyn,samples)
    
    # Panning among original signal and the low pass
    L0 = L**(1/3)
    samples = np.array(samples)
    samples = (L*L0*samples) + (1.0-L)*y
    return samples

"""
Original digitar KS damping filter, two-point avarage
The filter goes in the string feedback loop
"""
def original_KS_digitar_damping_filter(Fs,f0,D) :
    if VERBOSE:
        print("original_KS_digitar_damping_filter")
    
    rho = 1 #0.996
    g0 = 0.5*rho
    g1 = 0.5*rho
    
    N_lagrange = 4
    
    OldMax = 1
    OldMin = 0
    NewMin = N_lagrange-0.5
    NewMax = N_lagrange+0.5

    delta = Fs/f0 - int(Fs/f0)
    NewDelta = (((delta - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    print(delta)
    print(NewDelta)
    h_lagrange = lagrange(N_lagrange, NewDelta)
    coeff = (-1*np.convolve([g0,g1],h_lagrange)).tolist()
    
    coeff = [-g0,-g1]
    print(coeff)
    b = np.array([1.0]) # Zeros numerator coefficients
    #b = np.array(([0]*D) + h_lagrange)
    a = np.array([1.0] + ([0]*(D-1))  +(coeff)) # Poles denominator coefficients

    return b,a

"""
Original EKS damping filter, weighted two-point avarage
The filter goes in the string feedback loop
Hd(z) = (1-S) + S*z^(-1)
S is stretching factor, it adjusts the relative decay-rate for high versus low frequencies in the string
S = 0 or S = 1 the decay time is stretched infinitely (no decay)
S = 0.5 fastest decay, the filter reduce to the KS digitar damping filter (avaraging the last two samples of the delay line)
The decay rate is always infinity for DC, and higher frequencies decay faster than lower frequencies when S is in (0,1)
"""
def original_EKS_damping_filter(D,S) :
    if VERBOSE:
        print("original_EKS_damping_filter : "," S = ",S)
    
    rho = 0.996
    g0 = (1-S)*rho
    g1 = S*rho   
        
    b = np.array([1.0]) # Zeros numerator coefficients
    a = np.array([1.0] + ([0]*(D-3)) + [-g0, -g1]) # Poles denominator coefficients
    return b,a

"""
ONE-ZERO STRING damping FILTER
To control the overall decay rate, another (frequency independent) gain multiplier rho in (0,1) was introduced
to give the loop filter Hd(z) = rho[(1-S) + S*z^(-1)]
Since this filter is applied once per period P = (self.D) at the fundamental frequency, an attenuation of |Hd(e^(j*2*pi/P)| = rho
occours once each P samples. Setting rho to achieve a decay of -60 dB in t_60 seconds is obatained solving:
rho^(t_60/(P*T)) = 0.001 => rho = (0.001)^((P*T)/t_60)
note that P * T = f0*T = f0/f0 = 1

t_60 is the decay time of the string
B is a brightess parameter between 0 and 1
"""
def one_zero_damping_filter(f0,D,t_60=4,B=0.9) :
    if VERBOSE:
        print("one_zero_damping_filter : "," t_60 = ",t_60," B = ",B)
        
    rho = 0.001**(1.0/(f0*t_60))
    h0 = 0.5*B # S = B/2
    h1 = 1-h0 # 1-S

    g0 = rho*h0
    g1 = rho*h1

    b = np.array([1.0]) # Zeros numerator coefficients
    a = np.array([1.0] + ([0]*(D-3)) + [-g0 , -g1]) # Poles denominator coefficients
    return b,a

"""
TWO-ZERO STRING damping FILTER
A disadvantage of the decay-stretching parameter is that it affects tuning, except when S = 0.
This can be alleviated by going to a second-order, symmetric, linear-phase FIR filter having a
transfer function of the form Hd(z) = g1 + g0*z^(-1) + g1*z^(-2) = z^(-1)*[g0 + g1*(z+z^(-1))]
Due to the symmetry of the impulse response hd = [g1,g0,g1,0,0,...] about time n=1, only two
multiplies and two additions are needed per sample. The previous one-zero loop-filter required
one multiply and two additions per sample. Since the delay is equal to one sample at all the
frequencies (in the needed conefficien range) we obatin tuning invariance for the price of one
additional multiply per sample. We also obatain a bit more lowpass filtering.
The one-zero loop filter has a "lighter-sweeter",  tone than the two-zero case. In general,
the tone is quite sensitive to the details of all filtering in the feedback path.

t_60 is the decay time of the string
B is a brightess parameter between 0 and 1
"""
def two_zero_damping_filter(f0,D,t_60=4,B=0.9) :
    if VERBOSE:
        print("two_zero_damping_filter : "," t_60 = ",t_60," B = ",B)
        
    rho = 0.001**(1.0/(f0*t_60))
    h0 = (1.0 + B)/2
    h1 = (1.0 - B)/4
    g0 = rho*h0
    g1 = rho*h1
    
    b = np.array([1.0]) # Zeros numerator coefficients
    a = np.array([1.0] + ([0]*(D-3)) + [-g1 , -g0, -g1]) # Poles denominator coefficients
    return b,a

def pick_string(f0, Fs, duration, **kwargs) :
    
    damping_filter_types = ["KS", "EKS", "one_zero", "two_zero"]
    
    D = int(Fs/f0)
    #D_frac = Fs/f0
    N = Fs*duration
    x = np.array([0]*N)
    
    """
    if kwargs is not None:
        for key, value in kwargs.items():
            print (key,value)
    """
    
    name = kwargs.get('name')
    if not name:
        name = "note"
    
    damping_filter = kwargs.get("damping_filter")
    if not damping_filter: 
        damping_filter = "EKS"
    else : 
        if not damping_filter in damping_filter_types :
            damping_filter = "EKS"
    
    if damping_filter == "KS":
        b,a = original_KS_digitar_damping_filter(Fs,f0,D) # Determine the coefficents of the damping filter
        
    elif damping_filter == "EKS":
        S = kwargs.get("S")
        if not S:
            S = 0.8 # S is stretching factor, it adjusts the relative decay-rate for high versus low frequencies in the string
            #S = 0.5 -> Original digitar KS damping filter, two-point avarage
        b,a = original_EKS_damping_filter(D,S) # Determine the coefficents of the damping filter
    
    else :
        t_60 = kwargs.get("t_60")
        if not t_60:
            t_60 = 4.0 # in seconds
            #t_60 = 0
            #t_60 = 10
            #t_60 = 0.01
        B = kwargs.get("B")
        if not B:
            B = 0 # B is a brightess parameter between 0 and 1
        
        if damping_filter == "one_zero":
            b,a = one_zero_damping_filter(f0,D,t_60,B) # Determine the coefficents of the damping filter
        elif damping_filter == "two_zero":
            b,a = two_zero_damping_filter(f0,D,t_60,B) # Determine the coefficents of the damping filter
    
    # Plot the frequency response of the damping filter
    """nfft = 4096
    F = linspace(0, 1000*2*math.pi/Fs, 4096);
    w,h = freqz(b,a,F)
    h_dB = 20 * log10(abs(h))
    plot(w/max(w),h_dB)
    ylabel('Magnitude (db)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Frequency response')
    show()"""
    ###
    
    """ FILTER THE EXCITATION, BEFORE LOOP STRING """
    # Initial state with random excitation
    delayline = excitation(max([len(a),len(b)]) - 1)
    
    pick_dir = kwargs.get("pick_direction")
    if pick_dir is None:
        pick_dir = True
    
    pick_pos = kwargs.get("pick_position")
    if pick_pos is None:
        pick_pos = True
    
    if pick_dir:
        angle = kwargs.get("angle")
        if not angle:
            # Choice of pick direction (angle)
            angle = 0.9 # values between 0 and 1
        delayline = pick_direction(delayline,angle)
    
    if pick_pos:
        position = kwargs.get("position")
        if not position: 
            # choice of pick position (bridge,nut)
            position = 0.0 # values between 0 and 1
        delayline = pick_position(delayline,D,position)
    
    """ FILTER IN THE LOOP STRING """
    # Loop-String applying the damping filter
        
    samples, delayline = lfilter(b,a,x,-1,delayline)
    #samples, delayline = karplus_strong_diff_eq(f0,Fs,duration)
    
    
    """ FILTER AFTER THE LOOP STRING """
    # Dynamic Level Low Pass Filter
    
    dynamic_level = kwargs.get("dynamic_level")
    if dynamic_level is None:
        dynamic_level = True
    
    if dynamic_level:
        L = kwargs.get("L")
        if not L:
            #L = 0
            #L = 0.001
            #L = 0.01
            L = 0.1
            #L = 0.2
            #L = 0.32 # max is 1/3
        samples = dynamic_level_lpfilter(f0,Fs,samples,L)
    
    return samples
    
    

def wave(Fs,samples,name="note") :
    scaled = np.int16(samples/np.max(np.abs(samples)) * 32767)
    write(name+'.wav', Fs, scaled)

def play_chord(Fs,chord_frets,duration,offset_strum = 500):
    A = 110; # The A string of a guitar is normally tuned to 110 Hz
    offset_strings = [-5, 0, 5, 10, 14, 19] # offset of frets for each string E A D G B e
    name_strings = ["E","A","D","G","B","E2"]
    nsamples = Fs*duration
    #frequencies = []
    #delays = []
    
    # 50 ms delay btw the strings
    delay_strum = round(offset_strum*Fs/1000)
    nsamples_strum = nsamples + 6*delay_strum
    
    chord = np.zeros(nsamples)
    chord_strum = np.zeros(nsamples_strum)
    
    i = 0
    for fret, offset, name_string in zip(chord_frets, offset_strings, name_strings):
        # Each fret along a guitar's neck is a half tone
        f = (A*2**((fret+offset)/12.0))
        #frequencies.append(f)
        #delays.append(round(Fs/f))
        
        #samples = pick_string(f, Fs, duration, name = name_string, damping_filter = "one_zero",t_60=4)
        samples = pick_string(f, Fs, duration, name = name_string, damping_filter = "two_zero",t_60=4,B=0.5,angle=0.9,position=0.5,L=0.1)
        #samples = pick_string(f, Fs, duration, name = name_string, damping_filter = "KS", dynamic_level=False, pick_direction=False, pick_position=False)
        
        if VERBOSE:
            print("")
    
        chord = chord + samples
        chord_strum[i*delay_strum:(i*delay_strum)+nsamples] = chord_strum[i*delay_strum:(i*delay_strum)+nsamples] + samples
        i+=1
    return chord_strum
    

def main():
    Fs = 16000
    duration = 4
    
    #name_string = "A"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "KS",dynamic_level=False, pick_direction=False, pick_position=False)
    
    #name_string = "A_pickdirection_angle05"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "KS", angle = 0.5, dynamic_level=False, pick_position=False)
    
    #name_string = "A_pickdirection_angle01"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "KS", angle = 0.1, dynamic_level=False, pick_position=False)
    
    #name_string = "A_pickdirection_angle09_position0"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "KS", angle = 0.9, position = 0, dynamic_level=False)
    
    #name_string = "A_pickdirection_angle09_position1"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "KS", angle = 0.9, position = 1.0, dynamic_level=False)
    
    #name_string = "A_pickdirection_angle09_position05"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "KS", angle = 0.9, position = 0.5, dynamic_level=False)
    
    #name_string = "A_pickdirection_angle09_position06"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "KS", angle = 0.9, position = 0.6, )
    
    #name_string = "A_pickdirection_angle09_position05_one_zero_t60_4_B1"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "one_zero", t_60 = 4, B=1, angle = 0.9, position = 0.5, dynamic_level=False)
    
    #name_string = "A_pickdirection_angle09_position05_one_zero_t60_10_B05"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "one_zero", t_60 = 10, B=0.5, angle = 0.9, position = 0.5, dynamic_level=False)
    
    #name_string = "A_pickdirection_angle09_position05_two_zero_t60_4_B05"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "two_zero", t_60 = 4, B=0.5, angle = 0.9, position = 0.5, dynamic_level=False)
    
    #name_string = "A_pickdirection_angle09_position05_two_zero_t60_10_B05"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "two_zero", t_60 = 10, B=0.5, angle = 0.9, position = 0.5, dynamic_level=False)
    
    
    #name_string = "A_pickdirection_angle09_position05_one_zero_t60_4_B05_L02"
    #name_string = "A_NOT_interpolated"
    #A = pick_string(110, Fs = 16000, duration = 4, name = name_string, damping_filter = "one_zero", t_60 = 4, B=0.5, angle = 0.9, position = 0.5, L = 0.2)
    #print(A)
    #name_string = "A_pickdirection_angle09_position05_one_zero_t60_4_B1"
    #A = pick_string(110, Fs = 44100, duration = 4, name = name_string, damping_filter = "one_zero", t_60 = 4, B=1, angle = 0.9, position = 0.5, L = 0.1)
    
    name_string = "A_digitar_not_interpolated"
    A = pick_string(110, Fs, duration = 4, name = name_string, damping_filter = "KS",dynamic_level=False, pick_direction=False, pick_position=False)
    
    wave(Fs,A,name_string)
    exit()
    
    if len(sys.argv) != 7 :
        print("You have to type the tab for the chord that you want to play. Ex: Karplus-Strong.py 3 3 2 0 1 3")
        exit()
    chord_frets = []
    for i in range(len(sys.argv)) :
        if i > 0 :
            chord_frets.append(int(sys.argv[i]))
    
    chord_strum = play_chord(Fs, chord_frets, duration,offset_strum = 500)
    wave(Fs,chord_strum,'chord_strum.wav')


if __name__ == '__main__':
    main()