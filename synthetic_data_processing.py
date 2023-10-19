#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:08:29 2023

@author: Levin Stiehl

By superimposing noise traces on seismic traces I produce a new synthetic dataset and
compare the SNR, maximum amplitude and cc improvement with DeepDenoiser.

"""

import seisbench.models as sbm
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime, Stream
import numpy as np
from obspy.signal.cross_correlation import correlate

#%% Timestamps
st = read('/data2/chile/CHILE_COMBINED_2021/2013/CX/PB11/HH*.D/CX.PB11..HH*.D.2013.001')
#st.filter("highpass", freq=1.0)
st.filter("highpass", freq=0.2)

# Timestemps for events and noise

#events
dt2 = UTCDateTime('2013-01-01 00:17:55.1984Z')
dt3 = UTCDateTime('2013-01-01 01:20:31.4102Z')
dt4 = UTCDateTime('2013-01-01 13:52:05.1328Z')
dt5 = UTCDateTime('2013-01-01 00:51:07.8856Z')
dt6 = UTCDateTime('2013-01-01 03:51:32.0654Z')
dt7 = UTCDateTime('2013-01-01 04:13:27.9533Z')
dt8 = UTCDateTime('2013-01-01 08:32:32.3587Z')
dt9 = UTCDateTime('2013-01-01 09:56:54.4998Z')
dt10 = UTCDateTime('2013-01-01 10:07:49.3953Z')
dt11 = UTCDateTime('2013-01-01 16:29:07.5167Z')
dt12 = UTCDateTime('2013-01-01 17:05:45.9022Z')

#noise
no_str = 2
dtn1 = UTCDateTime('2013-01-01 00:16:00.7704Z')
dtn3 = UTCDateTime('2013-01-01 09:35:26.5128Z') 
dtn5 = UTCDateTime('2013-01-01 12:16:09.8813Z') 
dtn6 = UTCDateTime('2013-01-01 13:24:00.1143Z') 
dtn7 = UTCDateTime('2013-01-01 13:39:00.6252Z') 
dtn8 = UTCDateTime('2013-01-01 16:25:54.9731Z')
dtn10 = UTCDateTime('2013-01-01 18:29:30.2010Z') 
dtn11 = UTCDateTime('2013-01-01 19:13:59.9461Z') 
dtn12 = UTCDateTime('2013-01-01 22:19:00.1419Z') 
# Abweichung der Snr Kontrolle durch ungleiche Verteilung von std innerhalb des Nois Signals

#%% DeepDenoiser

model = sbm.DeepDenoiser.from_pretrained("original")
#Originally published under CC0 1.0. Original available at https://doi.org/10.7910/DVN/YTMYO9 
#.\n\nConverted to SeisBench by Jannes Münchmeyer (munchmej@gfz-potsdam.de)'}


def deepDenoiser(dt,dt2=30, snr=2, dtn=dtn1):
    """
  Applies denoising to a seismic signal within a specified time window.
  
  Args:
      dt (obspy.UTCDateTime): Start time for the signal extraction.
      dt2 (int, optional): Duration of the signal extraction in seconds. Default is 30 seconds.
      snr (float, optional): Signal-to-noise ratio (SNR) for noise addition. Default is 2.
      dtn (obspy.UTCDateTime): Start time for the noise signal extraction.
  
  Returns:
      annotations (obspy.Stream): Stream with annotations after denoising.
      ste (obspy.Stream): Original seismic signal within the specified time window.
      noisy_data (obspy.Stream): Noisy signal with added noise.
  """
    
    #slice signal
    ste = st.slice(dt, dt + dt2) 
    #slice noise
    stn = st.slice(dtn, dtn + dt2)

    #add noise
    noisy_data = add_noise_to_stream(ste, stn, snr)
    #signal denoisen whith DeepDenoiser
    annotations = model.annotate(noisy_data)

    #plots results
    
    # plt.figure(figsize=(10,4))
    # plt.subplot(331)
    # plt.plot(ste[0], 'k', linewidth=0.5, label="E")
    # plt.legend()
    # plt.title("Raw signal")
    # plt.subplot(332)
    # plt.plot(noisy_data[0], 'k', linewidth=0.5, label="E")
    # plt.title("Noisy signal")
    # plt.subplot(333)
    # plt.plot(annotations[0], 'k', linewidth=0.5, label="E")
    # plt.title("Denoised signal")
    # plt.subplot(334)
    # plt.plot(ste[1], 'k', linewidth=0.5, label="N")
    # plt.legend()
    # plt.subplot(335)
    # plt.plot(noisy_data[1], 'k', linewidth=0.5, label="N")
    # plt.subplot(336)
    # plt.plot(annotations[1], 'k', linewidth=0.5, label="N")
    # plt.subplot(337)
    # plt.plot(ste[2], 'k', linewidth=0.5, label="Z")
    # plt.legend()
    # plt.subplot(338)
    # plt.plot(noisy_data[2], 'k', linewidth=0.5, label="Z")
    # plt.subplot(339)
    # plt.plot(annotations[2], 'k', linewidth=0.5, label="Z")
    # plt.tight_layout()
    # plt.show();
      
    return annotations, ste, noisy_data

#%%
def add_noise_to_stream(stream, noise, snr):
    """
   Adds noise to a seismic stream with the desired SNR.
   
   Args:
       stream (obspy.Stream): Original seismic stream.
       noise (obspy.Stream): Stream containing the noise signal.
       snr (float): Desired signal-to-noise ratio (SNR).
   
   Returns:
       noisy_stream (obspy.Stream): Stream with added noise based on the specified SNR.
   """
    
    # Überprüfen Sie, ob die Anzahl der Traces im Stream-Objekt und im Rauschen-Objekt übereinstimmen
    if len(stream) != len(noise):
        raise ValueError("Die Anzahl der Traces im Stream und im Rauschen muss übereinstimmen.")
    
    # Erstellen Sie eine leere Liste für die modifizierten Traces
    modified_traces = []
    
    # Fügen Sie das Rauschen zu jedem Trace im Stream-Objekt hinzu
    for trace, noise_trace in zip(stream, noise):
        # Überprüfen Sie, ob die Abtastrate der Traces übereinstimmt
        if trace.stats.sampling_rate != noise_trace.stats.sampling_rate:
            raise ValueError("Die Abtastrate der Traces muss übereinstimmen.")
        
        # Berechnen Sie die Skalierungsfaktoren basierend auf dem SNR und den Standardabweichungen
        std_signal = np.std(trace.data[500:])
        std_noise = np.std(noise_trace.data)
        print('std_noise',std_noise,'std_noise 500', np.std(noise_trace.data[:500]))
        # dB
        #std_target = std_signal / snr 
        std_target = std_signal / (10**(snr / 10))
        adjustment_factor = std_target / std_noise
        
        # Skalieren Sie das Rauschen basierend auf dem SNR
        scaled_noise = noise_trace.data * adjustment_factor
        
        calculate_snr(trace.data[500:], scaled_noise.data)
        
        # Fügen Sie das Rauschen zum Trace hinzu
        noisy_data = trace.data + scaled_noise
        noisy_trace = trace.copy()
        noisy_trace.data = noisy_data
        
        # Fügen Sie den modifizierten Trace zur Liste hinzu
        modified_traces.append(noisy_trace)
        
        # plt.figure(figsize=(10,4))
        # plt.subplot(331)
        # plt.plot(trace.data, 'k', linewidth=0.5, label="Z")
        # plt.legend()
        # plt.title("Raw signal")
        # plt.subplot(332)
        # plt.plot(scaled_noise, 'k', linewidth=0.5, label="Z")
        # plt.title("Noise")
        # plt.subplot(333)
        # plt.plot(noisy_data, 'k', linewidth=0.5, label="Z")
        # plt.title("Nosiy signal")
        # plt.tight_layout()
        # plt.show();
    
    # Erstellen Sie ein neues Stream-Objekt mit den modifizierten Traces
    noisy_stream = Stream(traces=modified_traces)
    
    
    return noisy_stream

#%%
def calculate_snr(signal, noise):
    """
   Calculates the signal-to-noise ratio (SNR) between a signal and noise.
   
   Args:
       signal (numpy.ndarray): Signal data array.
       noise (numpy.ndarray): Noise data array.
   
   Returns:
       snr (float): Signal-to-noise ratio (SNR) in dB.
   """
    
    # Berechnen Sie die Standardabweichungen des Signals und des Rauschens
    std_signal = np.std(signal, axis=0)
    std_noise = np.std(noise, axis=0)
    
    # Berechnen Sie das SNR in Dezibel (dB)
    snr = 10 * np.log10(std_signal / std_noise)
    #snr = std_signal / std_noise

    print(snr)
    
    return snr

#%%

def Cross_Corelation(deno_da,evt_da):
    
    """
    Computes the cross-correlation between two seismic data arrays.
    
    Args:
        deno_da (numpy.ndarray): Denoised seismic data array.
        evt_da (numpy.ndarray): Original seismic data array.
    
    Returns:
        corr (float): Maximum cross-correlation value at zero lag.
    """
    
    #cross Corelation zero lag machen
    corr = max(correlate(deno_da[500:], evt_da[500:], 0, normalize= 'naive')) #zero Lag
    #corr = max(correlate(deno_da, evt_da, 100, normalize= 'naive'))
    print(corr)
    
    
    # plt.figure(figsize=(10,7))
    # plt.subplot(211)
    # plt.plot(deno_da, 'k', linewidth=0.5, label="Z")
    # plt.legend()
    # plt.title("Denoised signal")
    # plt.subplot(212)
    # plt.plot(evt_da, 'k', linewidth=0.5, label=f'{corr}')
    # plt.title("Raw signal")
    # plt.legend()  
    # plt.show()
        
    return corr

#%% Grafiken erstellen

#note: disable plots in other funktions

snr_test = np.arange(0.1,13.1,0.3)
snr_test = np.flip(snr_test)

y_data_SNR_mean = []

y_data_CC_mean_deno = []
y_data_CC_mean_noisy = []

y_data_AC_mean_deno = []
y_data_AC_mean_noisy = []

y_data_CC_mean_deno2 = []
y_data_CC_mean_noisy2 = []

y_data_AC_mean_deno2 = []
y_data_AC_mean_noisy2 = []

stn_mögl = [dtn1, dtn3, dtn5, dtn6, dtn7, dtn8, dtn10, dtn11, dtn12]

#denoising Durschführen
deno02, evt02, noisy02 = deepDenoiser(dt2-5,snr=90)
deno03, evt03, noisy03 = deepDenoiser(dt3-5,snr=90)
deno04, evt04, noisy04 = deepDenoiser(dt4-5,snr=90)
deno05, evt05, noisy05 = deepDenoiser(dt5-5,snr=90)
deno06, evt06, noisy06 = deepDenoiser(dt6-5,snr=90)
deno07, evt07, noisy07 = deepDenoiser(dt7-5,snr=90)
deno08, evt08, noisy08 = deepDenoiser(dt8-5,snr=90)
deno09, evt09, noisy09 = deepDenoiser(dt9-5,snr=90)
deno010, evt010, noisy010 = deepDenoiser(dt10-5,snr=90)
deno011, evt011, noisy011 = deepDenoiser(dt11-5,snr=90)
deno012, evt012, noisy012 = deepDenoiser(dt12-5,snr=90)

for snr_vor in snr_test:
    
    snr_mittel_ar = []
    
    CC_mittel_noisy_ar = []
    CC_mittel_deno_ar = []
    
    AC_mittel_noisy_ar = []
    AC_mittel_deno_ar = []
    
    CC_mittel_noisy_ar2 = []
    CC_mittel_deno_ar2 = []
    
    AC_mittel_noisy_ar2 = []
    AC_mittel_deno_ar2 = []
    
    for dtn in stn_mögl:
        
        #denoising Durschführen
        deno2, evt2, noisy2 = deepDenoiser(dt2-5,snr=snr_vor,dtn=dtn)
        deno3, evt3, noisy3 = deepDenoiser(dt3-5,snr=snr_vor,dtn=dtn)
        deno4, evt4, noisy4 = deepDenoiser(dt4-5,snr=snr_vor,dtn=dtn)
        deno5, evt5, noisy5 = deepDenoiser(dt5-5,snr=snr_vor,dtn=dtn)
        deno6, evt6, noisy6 = deepDenoiser(dt6-5,snr=snr_vor,dtn=dtn)
        deno7, evt7, noisy7 = deepDenoiser(dt7-5,snr=snr_vor,dtn=dtn)
        deno8, evt8, noisy8 = deepDenoiser(dt8-5,snr=snr_vor,dtn=dtn)
        deno9, evt9, noisy9 = deepDenoiser(dt9-5,snr=snr_vor,dtn=dtn)
        deno10, evt10, noisy10 = deepDenoiser(dt10-5,snr=snr_vor,dtn=dtn)
        deno11, evt11, noisy11 = deepDenoiser(dt11-5,snr=snr_vor,dtn=dtn)
        deno12, evt12, noisy12 = deepDenoiser(dt12-5,snr=snr_vor,dtn=dtn)

        #SNR berechnen
        snr2 = calculate_snr(deno2[2].data[500:], deno2[2].data[:500])
        snr3 = calculate_snr(deno3[2].data[500:], deno3[2].data[:500])
        snr4 = calculate_snr(deno4[2].data[500:], deno4[2].data[:500])  
        snr5 = calculate_snr(deno5[2].data[500:], deno5[2].data[:500])
        snr6 = calculate_snr(deno6[2].data[500:], deno6[2].data[:500])
        snr7 = calculate_snr(deno7[2].data[500:], deno7[2].data[:500])
        snr8 = calculate_snr(deno8[2].data[500:], deno8[2].data[:500])
        snr9 = calculate_snr(deno9[2].data[500:], deno9[2].data[:500])
        snr10 = calculate_snr(deno10[2].data[500:], deno10[2].data[:500])
        snr11 = calculate_snr(deno11[2].data[500:], deno11[2].data[:500])
        snr12 = calculate_snr(deno12[2].data[500:], deno12[2].data[:500])
    
        snr_mittel = np.mean([snr2,snr3,snr4,snr5,snr6,snr7,snr8,snr9,snr10,snr11,snr12])
        snr_mittel_ar.append(snr_mittel)
        
        #Cross Corelation berechnen
        corr2 = Cross_Corelation(noisy2[2].data, evt2[2].data)
        corr3 = Cross_Corelation(noisy3[2].data, evt3[2].data)
        corr4 = Cross_Corelation(noisy4[2].data, evt4[2].data)
        corr5 = Cross_Corelation(noisy5[2].data, evt5[2].data)
        corr6 = Cross_Corelation(noisy6[2].data, evt6[2].data)
        corr7 = Cross_Corelation(noisy7[2].data, evt7[2].data)
        corr8 = Cross_Corelation(noisy8[2].data, evt8[2].data)
        corr9 = Cross_Corelation(noisy9[2].data, evt9[2].data)
        corr10 = Cross_Corelation(noisy10[2].data, evt10[2].data)
        corr11 = Cross_Corelation(noisy11[2].data, evt11[2].data)
        corr12 = Cross_Corelation(noisy12[2].data, evt12[2].data)
        
        CC_mittel_noisy = np.mean([corr2,corr3,corr4,corr5,corr6,corr7,corr8,corr9,corr10,corr11,corr12])
        CC_mittel_noisy_ar.append(CC_mittel_noisy)
        
        corr2 = Cross_Corelation(deno2[2].data, evt2[2].data)
        corr3 = Cross_Corelation(deno3[2].data, evt3[2].data)
        corr4 = Cross_Corelation(deno4[2].data, evt4[2].data)
        corr5 = Cross_Corelation(deno5[2].data, evt5[2].data)
        corr6 = Cross_Corelation(deno6[2].data, evt6[2].data)
        corr7 = Cross_Corelation(deno7[2].data, evt7[2].data)
        corr8 = Cross_Corelation(deno8[2].data, evt8[2].data)
        corr9 = Cross_Corelation(deno9[2].data, evt9[2].data)
        corr10 = Cross_Corelation(deno10[2].data, evt10[2].data)
        corr11 = Cross_Corelation(deno11[2].data, evt11[2].data)
        corr12 = Cross_Corelation(deno12[2].data, evt12[2].data)
        
        
        CC_mittel_deno = np.mean([corr2,corr3,corr4,corr5,corr6,corr7,corr8,corr9,corr10,corr11,corr12])
        CC_mittel_deno_ar.append(CC_mittel_deno)
        
        #Max amplitud change normalizierd
        achange2 = (np.amax(np.abs(noisy2[2].data)) - np.amax(np.abs(evt2[2].data))) / np.amax(np.abs(evt2[2].data))
        achange3 = (np.amax(np.abs(noisy3[2].data)) - np.amax(np.abs(evt3[2].data))) / np.amax(np.abs(evt3[2].data))
        achange4 = (np.amax(np.abs(noisy4[2].data)) - np.amax(np.abs(evt4[2].data))) / np.amax(np.abs(evt4[2].data))
        achange5 = (np.amax(np.abs(noisy5[2].data)) - np.amax(np.abs(evt5[2].data))) / np.amax(np.abs(evt5[2].data))
        achange6 = (np.amax(np.abs(noisy6[2].data)) - np.amax(np.abs(evt6[2].data))) / np.amax(np.abs(evt6[2].data))
        achange7 = (np.amax(np.abs(noisy7[2].data)) - np.amax(np.abs(evt7[2].data))) / np.amax(np.abs(evt7[2].data))
        achange8 = (np.amax(np.abs(noisy8[2].data)) - np.amax(np.abs(evt8[2].data))) / np.amax(np.abs(evt8[2].data))
        achange9 = (np.amax(np.abs(noisy9[2].data)) - np.amax(np.abs(evt9[2].data))) / np.amax(np.abs(evt9[2].data))
        achange10 = (np.amax(np.abs(noisy10[2].data)) - np.amax(np.abs(evt10[2].data))) / np.amax(np.abs(evt10[2].data))
        achange11 = (np.amax(np.abs(noisy11[2].data)) - np.amax(np.abs(evt11[2].data))) / np.amax(np.abs(evt11[2].data))
        achange12 = (np.amax(np.abs(noisy12[2].data)) - np.amax(np.abs(evt12[2].data))) / np.amax(np.abs(evt12[2].data))

        AC_mittel_noisy = np.mean([achange2,achange3,achange4,achange5,achange6,achange7,achange8,achange9,achange10,achange11,achange12])
        AC_mittel_noisy_ar.append(AC_mittel_noisy*100)
        
        achange2 = (np.amax(np.abs(deno2[2].data)) - np.amax(np.abs(evt2[2].data))) / np.amax(np.abs(evt2[2].data))
        achange3 = (np.amax(np.abs(deno3[2].data)) - np.amax(np.abs(evt3[2].data))) / np.amax(np.abs(evt3[2].data))
        achange4 = (np.amax(np.abs(deno4[2].data)) - np.amax(np.abs(evt4[2].data))) / np.amax(np.abs(evt4[2].data))
        achange5 = (np.amax(np.abs(deno5[2].data)) - np.amax(np.abs(evt5[2].data))) / np.amax(np.abs(evt5[2].data))
        achange6 = (np.amax(np.abs(deno6[2].data)) - np.amax(np.abs(evt6[2].data))) / np.amax(np.abs(evt6[2].data))
        achange7 = (np.amax(np.abs(deno7[2].data)) - np.amax(np.abs(evt7[2].data))) / np.amax(np.abs(evt7[2].data))
        achange8 = (np.amax(np.abs(deno8[2].data)) - np.amax(np.abs(evt8[2].data))) / np.amax(np.abs(evt8[2].data))
        achange9 = (np.amax(np.abs(deno9[2].data)) - np.amax(np.abs(evt9[2].data))) / np.amax(np.abs(evt9[2].data))
        achange10 = (np.amax(np.abs(deno10[2].data)) - np.amax(np.abs(evt10[2].data))) / np.amax(np.abs(evt10[2].data))
        achange11 = (np.amax(np.abs(deno11[2].data)) - np.amax(np.abs(evt11[2].data))) / np.amax(np.abs(evt11[2].data))
        achange12 = (np.amax(np.abs(deno12[2].data)) - np.amax(np.abs(evt12[2].data))) / np.amax(np.abs(evt12[2].data))

        AC_mittel_deno = np.mean([achange2,achange3,achange4,achange5,achange6,achange7,achange8,achange9,achange10,achange11,achange12])
        AC_mittel_deno_ar.append(AC_mittel_deno*100)
        
        #Cross Corelation berechnen
        corr2 = Cross_Corelation(noisy2[2].data, deno02[2].data)
        corr3 = Cross_Corelation(noisy3[2].data, deno03[2].data)
        corr4 = Cross_Corelation(noisy4[2].data, deno04[2].data)
        corr5 = Cross_Corelation(noisy5[2].data, deno05[2].data)
        corr6 = Cross_Corelation(noisy6[2].data, deno06[2].data)
        corr7 = Cross_Corelation(noisy7[2].data, deno07[2].data)
        corr8 = Cross_Corelation(noisy8[2].data, deno08[2].data)
        corr9 = Cross_Corelation(noisy9[2].data, deno09[2].data)
        corr10 = Cross_Corelation(noisy10[2].data, deno010[2].data)
        corr11 = Cross_Corelation(noisy11[2].data, deno011[2].data)
        corr12 = Cross_Corelation(noisy12[2].data, deno012[2].data)
        
        CC_mittel_noisy = np.mean([corr2,corr3,corr4,corr5,corr6,corr7,corr8,corr9,corr10,corr11,corr12])
        CC_mittel_noisy_ar2.append(CC_mittel_noisy)
        
        corr2 = Cross_Corelation(deno2[2].data, deno02[2].data)
        corr3 = Cross_Corelation(deno3[2].data, deno03[2].data)
        corr4 = Cross_Corelation(deno4[2].data, deno04[2].data)
        corr5 = Cross_Corelation(deno5[2].data, deno05[2].data)
        corr6 = Cross_Corelation(deno6[2].data, deno06[2].data)
        corr7 = Cross_Corelation(deno7[2].data, deno07[2].data)
        corr8 = Cross_Corelation(deno8[2].data, deno08[2].data)
        corr9 = Cross_Corelation(deno9[2].data, deno09[2].data)
        corr10 = Cross_Corelation(deno10[2].data, deno010[2].data)
        corr11 = Cross_Corelation(deno11[2].data, deno011[2].data)
        corr12 = Cross_Corelation(deno12[2].data, deno012[2].data)       
        
        CC_mittel_deno = np.mean([corr2,corr3,corr4,corr5,corr6,corr7,corr8,corr9,corr10,corr11,corr12])
        CC_mittel_deno_ar2.append(CC_mittel_deno)
        
        #Max amplitud change normalizierd
        achange2 = (np.amax(np.abs(noisy2[2].data)) - np.amax(np.abs(deno02[2].data))) / np.amax(np.abs(deno02[2].data))
        achange3 = (np.amax(np.abs(noisy3[2].data)) - np.amax(np.abs(deno03[2].data))) / np.amax(np.abs(deno03[2].data))
        achange4 = (np.amax(np.abs(noisy4[2].data)) - np.amax(np.abs(deno04[2].data))) / np.amax(np.abs(deno04[2].data))
        achange5 = (np.amax(np.abs(noisy5[2].data)) - np.amax(np.abs(deno05[2].data))) / np.amax(np.abs(deno05[2].data))
        achange6 = (np.amax(np.abs(noisy6[2].data)) - np.amax(np.abs(deno06[2].data))) / np.amax(np.abs(deno06[2].data))
        achange7 = (np.amax(np.abs(noisy7[2].data)) - np.amax(np.abs(deno07[2].data))) / np.amax(np.abs(deno07[2].data))
        achange8 = (np.amax(np.abs(noisy8[2].data)) - np.amax(np.abs(deno08[2].data))) / np.amax(np.abs(deno08[2].data))
        achange9 = (np.amax(np.abs(noisy9[2].data)) - np.amax(np.abs(deno09[2].data))) / np.amax(np.abs(deno09[2].data))
        achange10 = (np.amax(np.abs(noisy10[2].data)) - np.amax(np.abs(deno010[2].data))) / np.amax(np.abs(deno010[2].data))
        achange11 = (np.amax(np.abs(noisy11[2].data)) - np.amax(np.abs(deno011[2].data))) / np.amax(np.abs(deno011[2].data))
        achange12 = (np.amax(np.abs(noisy12[2].data)) - np.amax(np.abs(deno012[2].data))) / np.amax(np.abs(deno012[2].data))

        AC_mittel_noisy = np.mean([achange2,achange3,achange4,achange5,achange6,achange7,achange8,achange9,achange10,achange11,achange12])
        AC_mittel_noisy_ar2.append(AC_mittel_noisy*100)
        
        achange2 = (np.amax(np.abs(deno2[2].data)) - np.amax(np.abs(deno02[2].data))) / np.amax(np.abs(deno02[2].data))
        achange3 = (np.amax(np.abs(deno3[2].data)) - np.amax(np.abs(deno03[2].data))) / np.amax(np.abs(deno03[2].data))
        achange4 = (np.amax(np.abs(deno4[2].data)) - np.amax(np.abs(deno04[2].data))) / np.amax(np.abs(deno04[2].data))
        achange5 = (np.amax(np.abs(deno5[2].data)) - np.amax(np.abs(deno05[2].data))) / np.amax(np.abs(deno05[2].data))
        achange6 = (np.amax(np.abs(deno6[2].data)) - np.amax(np.abs(deno06[2].data))) / np.amax(np.abs(deno06[2].data))
        achange7 = (np.amax(np.abs(deno7[2].data)) - np.amax(np.abs(deno07[2].data))) / np.amax(np.abs(deno07[2].data))
        achange8 = (np.amax(np.abs(deno8[2].data)) - np.amax(np.abs(deno08[2].data))) / np.amax(np.abs(deno08[2].data))
        achange9 = (np.amax(np.abs(deno9[2].data)) - np.amax(np.abs(deno09[2].data))) / np.amax(np.abs(deno09[2].data))
        achange10 = (np.amax(np.abs(deno10[2].data)) - np.amax(np.abs(deno010[2].data))) / np.amax(np.abs(deno010[2].data))
        achange11 = (np.amax(np.abs(deno11[2].data)) - np.amax(np.abs(deno011[2].data))) / np.amax(np.abs(deno011[2].data))
        achange12 = (np.amax(np.abs(deno12[2].data)) - np.amax(np.abs(deno012[2].data))) / np.amax(np.abs(deno012[2].data))

        AC_mittel_deno = np.mean([achange2,achange3,achange4,achange5,achange6,achange7,achange8,achange9,achange10,achange11,achange12])
        AC_mittel_deno_ar2.append(AC_mittel_deno*100)
    
    y_data_SNR_mean.append(np.mean(snr_mittel_ar))
    
    y_data_CC_mean_deno.append(np.mean(CC_mittel_deno_ar))
    y_data_CC_mean_noisy.append(np.mean(CC_mittel_noisy_ar)) 

    y_data_AC_mean_deno.append(np.mean(AC_mittel_deno_ar)) 
    y_data_AC_mean_noisy.append(np.mean(AC_mittel_noisy_ar)) 
    
    y_data_CC_mean_deno2.append(np.mean(CC_mittel_deno_ar2))
    y_data_CC_mean_noisy2.append(np.mean(CC_mittel_noisy_ar2)) 

    y_data_AC_mean_deno2.append(np.mean(AC_mittel_deno_ar2)) 
    y_data_AC_mean_noisy2.append(np.mean(AC_mittel_noisy_ar2)) 
  
#SNR Plot
plt.figure(figsize=(10,4))
plt.plot(snr_test, y_data_SNR_mean, 'r', marker='+', linewidth=0.5, label="DeepDenoiser")
plt.plot(snr_test, snr_test, 'b', marker='o', linewidth=0.5, label="Noisy signal", markersize=4)
plt.legend()
plt.xlabel('SNR before denoising (dB)')
plt.ylabel('SNR after denoising (dB)')
plt.title('Improvement of SNR')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()    

#Cross Correlation Plot
plt.figure(figsize=(10,4))
plt.plot(snr_test, y_data_CC_mean_deno, 'r', marker='+', linewidth=0.5, label="DeepDenoiser")
plt.plot(snr_test, y_data_CC_mean_noisy, 'b', marker='+', linewidth=0.5, label="Noisy signal")
plt.plot(snr_test, y_data_CC_mean_deno2, 'r', marker='o', linewidth=0.5, label="DeepDenoiser2", markersize=4)
plt.plot(snr_test, y_data_CC_mean_noisy2, 'b', marker='o', linewidth=0.5, label="Noisy signal2", markersize=4)
plt.legend()
plt.xlabel('SNR before denoising (dB)')
plt.ylabel('Correlation coefficient')
plt.title('Improvement of Correlation coefficient')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

#Amplitude change Plot
plt.figure(figsize=(10,4))
plt.plot(snr_test, y_data_AC_mean_deno, 'r', marker='+', linewidth=0.5, label="DeepDenoiser")
plt.plot(snr_test, y_data_AC_mean_noisy, 'b', marker='+', linewidth=0.5, label="Noisy signal")
plt.plot(snr_test, y_data_AC_mean_deno2, 'r', marker='o', linewidth=0.5, label="DeepDenoiser2", markersize=4)
plt.plot(snr_test, y_data_AC_mean_noisy2, 'b', marker='o', linewidth=0.5, label="Noisy signal2", markersize=4)
plt.legend()
plt.xlabel('SNR before denoising (dB)')
plt.ylabel('Max amplitude change (%)')
plt.title('Improvement of max amplitude')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()   
    


