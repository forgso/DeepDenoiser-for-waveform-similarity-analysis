#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:48:17 2022

@author: Levin Stiehl

DeepDenoiser
will be applied to events classified as RE. The cc different for RES at several
stations of different distance is calculated. Creating a dataset of cross-correlation coefficient
in relation to distance.

"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Stream, UTCDateTime
from obspy.signal.cross_correlation import correlate, xcorr_max, correlate_template
import os
import seisbench.models as sbm
from geopy.distance import distance
import pandas as pd
import csv
from scipy.optimize import curve_fit

#%%funktionen

def calculate_distance(lat1, lon1, depth1, lat2, lon2, depth2):
    """
    Calculate the three-dimensional distance between two geographical points
    (latitude, longitude, and depth).

    Parameters:
    lat1 (float): Latitude of the first point.
    lon1 (float): Longitude of the first point.
    depth1 (float): Depth of the first point.
    lat2 (float): Latitude of the second point.
    lon2 (float): Longitude of the second point.
    depth2 (float): Depth of the second point.

    Returns:
    total_distance (float): The three-dimensional distance between the two points.
    """
    station = (lat1, lon1)
    event = (lat2, lon2)

    flat_distance = distance(station, event).km
    
    # Consider the depth
    depth_distance = abs(depth2 - depth1)

    # Calculate the total distance (along the surface and in depth)
    total_distance = np.sqrt(flat_distance**2 + depth_distance**2)

    return total_distance
    
def loadKor_station(fn):
    """
    Load station latitude and longitude data from a file.

    Parameters:
    fn (str): The filename from which to load the data.

    Returns:
    station_lat (dict): A dictionary containing station codes as keys and their latitudes as values.
    station_lon (dict): A dictionary containing station codes as keys and their longitudes as values.
    """
    station_lat = dict()
    station_lon = dict()
    with open(fn, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            stat, lat, lon = line.split()
            station_lat.update({stat: float(lat)})
            station_lon.update({stat: float(lon)})

    return station_lat, station_lon
    
#%%Setup
#koordinaten Station berechnen
station_lon = dict()
station_lat = dict()

station_lat,station_lon = loadKor_station('/data/jonas/shared/chile_data_levin/CXstatlist.txt')

fn_REQS_all = '/data/jonas/shared/chile_data_levin/RE_all/REQS/REQS_all.csv'

df = pd.read_csv(fn_REQS_all)

# Öffne die CSV-Datei zum Lesen
with open(fn_REQS_all, 'r') as file:
    # CSV-Datei lesen
    reader = csv.reader(file)
    next(reader)  # Überspringe die Header-Zeile, falls vorhanden

    # Initialisiere ein leeres Dictionary zum Zählen der Vorkommen jeder cID
    cid_counts = {}

    # Durchlaufe die Zeilen der CSV-Datei
    for row in reader:
        cID = row[5]

        # Zähle die Vorkommen der cID
        if cID in cid_counts:
            cid_counts[cID] += 1
        else:
            cid_counts[cID] = 1

    # Initialisiere eine leere Liste für die cIDs mit 10-30 Vorkommen
    selected_cIDs = []

    # Durchlaufe das Dictionary der cID-Zählungen
    for cID, count in cid_counts.items():
        if 10 <= count <= 30:
            selected_cIDs.append(cID)

# Ausgabe der ausgewählten cIDs
print(selected_cIDs)

pick_lon = dict()#SN:median lon
pick_lat = dict()
pick_dep = dict()

for cID in selected_cIDs:
    # Filtere den DataFrame nach der gewünschten ID
    filtered_df = df.loc[df['cID'] == int(cID)]
    
    # Filtere die Latitudes und Longitudes nach der Bedingung
    filtered_latitudes = filtered_df['lat'].loc[filtered_df['lat'].apply(lambda x: len(str(x).split('.')[0]) == 3)]
    filtered_longitudes = filtered_df['lon'].loc[filtered_df['lon'].apply(lambda x: len(str(x).split('.')[0]) == 3)]
    filtered_depth = filtered_df['dep'].median()

    # Überprüfe, ob Werte vorhanden sind
    if not filtered_latitudes.empty and not filtered_longitudes.empty:
        # Berechne den Median der Latitudes und Longitudes
        median_latitude = filtered_latitudes.median()
        median_longitude = filtered_longitudes.median()
        
        pick_lon[cID] = median_longitude
        pick_lat[cID] = median_latitude   
        pick_dep[cID] = filtered_depth       
    else:
        print("Keine passenden Werte für ID:", cID)

#DataFrame CC all
df_CC = pd.DataFrame()
df_CC_deno = pd.DataFrame()
#%%start Analyse

#seriennummer   810 73
SN_REQS = [73]
for SN in selected_cIDs:

#for SN in SN_REQS:
    print(SN)

    path = '/data/jonas/shared/chile_data_levin/RE_data/'
    SEQdir = 'REQS_'+ str(SN) +'_wfs/'
    dir_entries = os.listdir(path + SEQdir)
    
    # make save directory if not there
    dirname = 'REQS_'+str(SN)+'_plots'
    if not(os.path.isdir(path + dirname)):
        try: 
            os.mkdir(path + dirname)
        except OSError as error: 
            print(error)
            
    # make save directory if not there
    dirnamedeno = 'REQS_'+str(SN)+'_plots_deno'
    if not(os.path.isdir(path + dirnamedeno)):
        try: 
            os.mkdir(path + dirnamedeno)
        except OSError as error: 
            print(error)
    
    statset = set()
    evset = set()
    for line in dir_entries:
        if 'HHZ'in line:
            statset.add(line.split('.')[0])
            evset.add(line.split('.')[2])
    
    evlist = sorted(list(evset))
    statlist  = sorted(list(statset))
    stat_reihnfolge = list()#reihnfolge der Stationen im array arraylist
    
    arraylist  = list()#CC
    arraylist_DeepDenoiser = list()#CC DeepDenoiser
    arraylist_dis  = dict()#Entfernung
    
    short_stream_speicher_raw = dict()#stat:stream
    short_stream_speicher_DD = dict()#stat:stream
    # auto picker einstellungen
    eqt_model = sbm.EQTransformer.from_pretrained("instance")   
    DD_model = sbm.DeepDenoiser.from_pretrained("original")
    
    #master event
    
    #dict mit allen datennamen und daten 
    mseedlist_dict_stat = dict()
    my_stream_dict_stat = dict()
    mseedlist_dict_evt = dict()

    #daten laden nach event gruppen und master event finden
    for evt in evlist:
        # Finde Filenamen        
        mseedlist_evt = []
        for s in dir_entries:
            # Bedingung
            if evt in s and 'HHZ' in s:
                mseedlist_evt.append(s)
            
        # sortiere die liste alphabetisch
        mseedlist_evt.sort()

        mseedlist_dict_evt[evt] = mseedlist_evt        
        
    master_event = max(mseedlist_dict_evt, key=lambda k: len(mseedlist_dict_evt[k]))
    
    pickdict = dict()#stat: peaktime
    #DeepDenoiser Loop 
    for n in [0,1]:
        print(n)
        
        #für masterevent an jeder station ein pick setzen
        
        for mseed_master in mseedlist_dict_evt[master_event]:
            master_stream = read (path + '/REQS_'+str(SN)+'_wfs/' + mseed_master)
            
            pick, detections = eqt_model.classify(master_stream)
            master_stream = master_stream.filter("bandpass", freqmin=1, freqmax=10, zerophase=True) 
            master_stream = master_stream.merge()
            #DeppDenoiser
            if n == 1:
                master_stream = DD_model.annotate(master_stream)
            
            master_trace = master_stream[0]
            
            #zweiten master trace referenz finden
            
            mseed_trace_ref = None
            stat = mseed_master.split('.')[0]
            
            for mseed in dir_entries:
                # Überprüfen, ob der Dateiname die gleiche Station wie das Masterevent hat, aber nicht das Masterevent selbst ist
                if stat in mseed and mseed != mseed_master and 'HHZ' in mseed:
                    mseed_trace_ref = mseed
                        
            # Überprüfen, ob ein übereinstimmender Dateiname gefunden wurde
            if mseed_trace_ref is not None:
                print("mseed_trace_ref")
            else:
                print("Keine master trace referenz gefunden.")
                continue
                
            ref_stream = read(path + '/REQS_'+str(SN)+'_wfs/' + mseed_trace_ref)
            
            master_stream_ref = ref_stream.filter("bandpass", freqmin=1, freqmax=10, zerophase=True) 
            master_stream_ref = ref_stream.merge()
            #DeppDenoiser
            if n == 1:
                ref_stream = DD_model.annotate(ref_stream)
            
            ref_trace = ref_stream[0]
            
            #plot test master und picks
            fig, ax = plt.subplots()
            
            #picks
            for p in pick:
                t = abs(master_trace.times("utcdatetime")[0] - p.start_time)
                plt.axvline(x=t, color='red', linestyle='--')
            
            #trace
            ax.plot(master_trace.times(), master_trace.data, label=mseed_master,
                        alpha=0.7,linewidth=1)
            plt.legend()
            #plt.close()
            
            #plot test ref
            fig, ax = plt.subplots()
            
            #trace
            ax.plot(ref_trace.times(), ref_trace.data, label=mseed_trace_ref,
                        alpha=0.7,linewidth=1)
            plt.legend()
            #plt.close()
    
            #finden beste picks
            bvCC = 0
            pvIDX = None
            pv = 0
            bvCC2 = 0#negativ fall
            pvIDX2 = None
            pv2 = 0
            
            if n == 0:
                #CC abgleich um pick für master trace slice zu bestimmen
                for i, p in enumerate(pick):
                    if p.peak_value > 0 and p.phase == 'P':
                        master_trace_short = master_trace.slice(p.peak_time -2, p.peak_time+33)
                        #slice evt auf 0-33 ändern
                        cc = correlate_template(ref_trace.data, master_trace_short.data, normalize='full')
                        
                        shift, valueCC = xcorr_max(cc)
                        #test absolute werte 
                        valueCC = abs(valueCC)
                        print(valueCC,stat)
                        if valueCC > bvCC and valueCC > 0.4:
                            bvCC = valueCC
                            pvIDX = i
                            pv = p.peak_value
                            print("neuer pick")
                        if valueCC < 0 and valueCC < bvCC2:
                            bvCC2= valueCC
                            pvIDX2 = i
                            pv2 = p.peak_value
                            print("neuer negativer pick")  
                
                if bvCC <= 0.4:
                  pv = pv2
                  pvIDX = pvIDX2 
                
                #speichern
                pickdict[stat] = {pv: pvIDX}
            else:
                pv, pvIDX = list(pickdict[stat].items())[0]
                
            #falls pick vorhanden seimogramme erstellen
            if pvIDX is not None:
                master_trace_short = master_trace.slice(pick[pvIDX].peak_time -2, pick[pvIDX].peak_time+33)
                
                fig, ax = plt.subplots()
           
                ax.plot(master_trace_short.times(), master_trace_short.data, label=mseed_master,
                            alpha=0.7,linewidth=1)
                t = abs(master_trace_short.times("utcdatetime")[0] - pick[pvIDX].peak_time)
    
                plt.axvline(x=t, color='black', linestyle='--')
    
                plt.legend()
                #plt.close()
                
                if stat not in stat_reihnfolge:
                    stat_reihnfolge.append(stat)
                            
                # Finde Filenamen    
                mseedlist = []
                for s in dir_entries:
                    # Bedingung
                    if stat in s and 'HHZ' in s:
                        mseedlist.append(s)
                        
                mseedlist_dict_stat[stat] = mseedlist
                
                #alle daten laden
                my_stream = Stream()
                my_stream_names = []
                error_idx = []
                for idx_m, s in enumerate(mseedlist):
                    st = read (path + '/REQS_'+str(SN)+'_wfs/' + s)
                    # filter  
                    st = st.detrend()
                    st = st.filter("bandpass", freqmin=1, freqmax=10, zerophase=True)
                    #st = st.merge()
                    if len(st[0].data) < 10201:
                        error_idx.append(idx_m)
                        print('error my_stream trace to short')
                        continue
                    
                    #DeppDenoiser
                    if n == 1:
                        st = DD_model.annotate(st)
                    tr = st[0]
                    my_stream += tr
                    my_stream_names += [s]   
                
                #fehlerhafte spuren aus mseedlist löschen
                for err in error_idx:
                    mseedlist.pop(err)

                #stream slicen anhand Lag time  
                my_stream_short = Stream()
               
                for idx, tr in enumerate(my_stream):
                     # Cross-Korrelation durchführen
                     if len(tr.data) < len(master_trace_short.data):
                         mseedlist.pop(idx) 
                         
                         print('error data to short')
                         continue
                     #t = mseedlist[idx].split('.')[-2] 
 
                     cc = correlate_template(tr.data, master_trace_short.data, normalize='full')
                    
                     shift, valueCC = xcorr_max(cc)
                    
                     #print(s,t," Shift:", shift, "CC:", valueCC)# ausgabe in sampels
                    
                     center_time = tr.stats.starttime + (tr.stats.endtime - tr.stats.starttime) / 2
                     #print(center_time)
                     my_stream_short += tr.slice(center_time-17+(shift/100), center_time+18+(shift/100))                     
                    
                if my_stream_short.count() == 0:
                     continue
                       
                #plot test
                fig, ax = plt.subplots()
           
                for idx, tr in enumerate(my_stream_short):
               
                   ax.plot(tr.times(), tr.data/max(abs(tr.data)) + idx , label=str(t),
                           alpha=0.7,linewidth=1)
           
                   ax.yaxis.set_ticklabels([])
                       
                #plt.legend()
                plt.title('Seismogramm SN ' + str(SN) + ' Station ' + str(stat))
                plt.xlabel('time in s',fontsize=12)
                plt.ylabel('amplitude in counts', fontsize=12)
           
                fig = plt.gcf()
                fig.set_size_inches(10, 10, forward=True)
                flnm= 'seismogram_SN' + str(SN) + '_stat_' + str(stat)
                if n ==0:
                    fig.savefig(path+dirname+'/'+flnm,format='pdf')  
                    fig.savefig(path+dirname+'/'+flnm + '.png',format='png')
                else:
                    fig.savefig(path+dirnamedeno+'/'+flnm,format='pdf')  
                    fig.savefig(path+dirnamedeno+'/'+flnm + '.png',format='png')      
                #plt.close()
                
                # Rechne Kreuzkorrelationen
                mseedlistshort = [item.split('.')[2] for item in mseedlist]
                
                my_array = np.zeros((len(evlist), len(evlist)))
                my_array[:] = np.nan
                for idx in range(my_array.shape[0]):
                    for idy in range(my_array.shape[1]):
                        
                        if evlist[idx] in mseedlistshort and evlist[idy] in mseedlistshort:
                            tr_idx = mseedlistshort.index(evlist[idx])
                            tr_idy = mseedlistshort.index(evlist[idy])
                        
                            tr1 = my_stream_short[tr_idx]
                            tr2 = my_stream_short[tr_idy]
                            cc = correlate(tr1, tr2, 1000, normalize= 'naive')
                          
                            shift, value = xcorr_max(cc)
                            
                            if shift > 50:
                                #print(idx,idy, shift, value)
                                cc = correlate(tr2, tr1, 1000 , normalize= 'naive')
                              
                                shift, value = xcorr_max(cc)
                                 
                            my_array[idx, idy] = value   
                                
                if n == 0:
                    arraylist.append(my_array)
                else:
                    arraylist_DeepDenoiser.append(my_array)
                    
                #Distanc berrechnen
                distanz_dic = dict()
                dis = calculate_distance(float(station_lat[stat]), float(station_lon[stat]),
                                         0, pick_lat[str(SN)], pick_lon[str(SN)], pick_dep[str(SN)])
                arraylist_dis[stat] = dis
                
            else: 
                print('Warnung: cc < 0.4')
        
        #CC arrays plots stations loop zu ende
        # plot my stacked array
        #np.nanmedian
        if n == 0:
            arraystack = np.dstack(arraylist)
        else:
            arraystack = np.dstack(arraylist_DeepDenoiser)
        
        arrayavg = np.zeros(arraystack.shape[0:2])
        arrayavg[:]  = np.nan
        for idx in range(arraystack.shape[0]):
            for idy in range(arraystack.shape[1]):
                arrayavg[idx, idy] = np.nanmedian(arraystack[idx, idy, :])
                             
        fig, ax = plt.subplots()
        im = ax.pcolormesh(arrayavg)
        
        plt.title('SN ' + str(SN) + ' Median')
        #ax.set_xlabel('Zeit')
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(len(evlist)))
        ax.set_xticks([float(n)+0.5 for n in ax.get_xticks()])
        ax.set_xticklabels(evlist,  fontsize = 4)
        
        ax.set_yticks(range(len(evlist)))
        ax.set_yticks([float(n)+0.5 for n in ax.get_yticks()])
        ax.set_yticklabels(evlist, fontsize = 4)
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                  rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(evlist)):
            for j in range(len(evlist)):
                text = ax.text(j+0.5, i+0.5, round(arrayavg[i, j],2),
                                ha="center", va="center", color="w",fontsize = 5)
                
        #plt.show()
        plt.colorbar(im)
        
        fig.set_size_inches(7, 7, forward=True)
        flnm= 'crossplot_SN' + str(SN) + '_fullmedian_cc.pdf'
        if n == 0:
            fig.savefig(path+dirname+'/'+flnm,format='pdf')   
        else:
            fig.savefig(path+dirnamedeno+'/'+flnm,format='pdf')    
        #plt.close()
        
        # plot average cc values based only on best 3 or 5 stations
        
        arrayavg = np.zeros(arraystack.shape[0:2])
        arrayavg[:]  = np.nan
        for idx in range(arraystack.shape[0]):
            for idy in range(arraystack.shape[1]):
                
                vec = arraystack[idx, idy, :]
                vec = np.nan_to_num(vec, nan=0)
                vec.sort()
               
                #print(idx,idy, vec)
                #print(vec)
                arrayavg[idx, idy] = np.mean(vec[-3:])           
                
        fig, ax = plt.subplots()
        im = ax.pcolormesh(arrayavg)
        
        plt.title('SN ' + str(SN) + ' Average')
        #ax.set_xlabel('Zeit')
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(len(evlist)))
        ax.set_xticks([float(n)+0.5 for n in ax.get_xticks()])
        ax.set_xticklabels(evlist,  fontsize = 4)
        
        ax.set_yticks(range(len(evlist)))
        ax.set_yticks([float(n)+0.5 for n in ax.get_yticks()])
        ax.set_yticklabels(evlist, fontsize = 4)
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                  rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(evlist)):
            for j in range(len(evlist)):
                text = ax.text(j+0.5, i+0.5, round(arrayavg[i, j],2),
                                ha="center", va="center", color="w",fontsize = 5)
                
        #plt.show()
        plt.colorbar(im)
        
        fig.set_size_inches(7, 7, forward=True)
        flnm= 'crossplot_SN' + str(SN) + '_best3avgcc.pdf'
        if n == 0:
            fig.savefig(path+dirname+'/'+flnm,format='pdf')   
        else:
            fig.savefig(path+dirnamedeno+'/'+flnm,format='pdf')   
        #plt.close()
        
        # save as csv
        fn = 'REQS_'+str(SN)+'_ccMatrix.csv'
        df = pd.DataFrame(data = arrayavg, index=evlist, columns = evlist)   
        if n ==0:
            df.to_csv(path+dirname+'/'+fn, index = True, na_rep =np.nan, mode= 'w')
        else:
            df.to_csv(path+dirnamedeno+'/'+fn, index = True, na_rep =np.nan, mode= 'w')
    #CC change berechnen, DD loop Ende
    
    median_array = list()
    median_array_DeepDenoiser = list()
    x_data_dis = list()
    
    for ar in arraylist:
        median_array.append(np.nanmedian(ar.flatten(), axis=0))
    for ar in arraylist_DeepDenoiser:
        median_array_DeepDenoiser.append(np.nanmedian(ar.flatten(), axis=0))
    for stat in stat_reihnfolge:
        x_data_dis.append(arraylist_dis[stat])
    median_array_DeepDenoiser = np.array(median_array_DeepDenoiser)
    median_array = np.array(median_array)
    #nach distanz sortieren
    sorted_dict = sorted(arraylist_dis.items(), key=lambda x: x[1])
    sorted_stat = [item[0] for item in sorted_dict]
    
    #CC change plot
    CC_change_dict = dict()
    for i, (ar_deep, ar_raw) in enumerate(zip(arraylist_DeepDenoiser, arraylist)):
        CC_change = np.array(ar_deep) - np.array(ar_raw)  
        CC_change_dict[stat_reihnfolge[i]] = np.nanmedian(CC_change.flatten(), axis=0)

    CC_change_list = list()
    for stat in sorted_stat:
        CC_change_list.append(CC_change_dict[stat])
    
    df_CC.loc[SN, stat_reihnfolge] = median_array
    df_CC_deno.loc[SN, stat_reihnfolge] = median_array_DeepDenoiser
    df_CC.loc[str(SN)+'_dis', stat_reihnfolge] = x_data_dis
    df_CC_deno.loc[str(SN)+'_dis', stat_reihnfolge] = x_data_dis
      
    fig, ax = plt.subplots()
    #plt.figure(figsize=(10,7))
    plt.scatter(sorted_stat, CC_change_list, label="DeepDenoiser-raw")
    plt.title("CC Change SN " + str(SN))
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Stations')
    plt.ylabel('Correlation coefficient difference')
    fig.set_size_inches(10, 7, forward=True)
    #_median_detrend
    flnm= 'CC_Change_SN' + str(SN) + '.pdf'
    fig.savefig(path+dirnamedeno+'/'+flnm,format='pdf') 
    #plt.close()
    
    #Plots
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7, forward=True)
    plt.scatter(x_data_dis,median_array_DeepDenoiser, label="DeepDenoiser")
    plt.scatter(x_data_dis,median_array, label="raw")
    plt.title("CC SN " + str(SN))
    plt.xlabel('Distance in km')
    plt.ylabel('Correlation coefficient')
    plt.ylim([0, 1])
    plt.legend()
    flnm= 'CC_SN' + str(SN) + '.pdf'
    fig.savefig(path+dirnamedeno+'/'+flnm,format='pdf') 
    #plt.close()
    
#%%plots CC dist mit allen Serien
# save as csv
#df_CC.to_csv('/data/jonas/shared/chile_data_levin/Figures/REQS_df_CCV4.csv', index = True, na_rep =np.nan, mode= 'w')
#df_CC_deno.to_csv('/data/jonas/shared/chile_data_levin/Figures/REQS_df_CC_denoV4.csv', index = True, na_rep =np.nan, mode= 'w')

# load csv to df index col per hand nachtragen
df_CC = pd.read_csv('/data/jonas/shared/chile_data_levin/Figures/REQS_df_CCV3.csv', index_col="index")
df_CC_deno = pd.read_csv('/data/jonas/shared/chile_data_levin/Figures/REQS_df_CC_denoV3.csv', index_col="index")

CC_Change_y_data = list()
CC_y_data = list()
CC_deno_y_data = list()

dis_x_data = list()
#Daten aus df laden
for ind in selected_cIDs:
    for stat in statlist:
        CC_value = df_CC.loc[ind, stat] 
        CC_deno_value = df_CC_deno.loc[ind, stat] 
        dis_value = df_CC.loc[str(ind)+'_dis', stat] 
        if not np.isnan(CC_value) and not np.isnan(CC_deno_value) and not np.isnan(dis_value): 
            CC_change_value = CC_deno_value - CC_value
            CC_y_data.append(CC_value)
            CC_deno_y_data.append(CC_deno_value)
            dis_x_data.append(dis_value)
            CC_Change_y_data.append(CC_change_value)

CC_Change_y_data = np.array(CC_Change_y_data) 
dis_x_data = np.array(dis_x_data) 
CC_y_data = np.array(CC_y_data) 
CC_deno_y_data = np.array(CC_deno_y_data) 
 
#%% scatterplot

# Koeffizienten der linearen Regression berechnen (Grad 1)
def fit_ex(x, a, b, c):
    return c - b**(x*a) 

# Erzeugen von x-Werten für den Plot
x_fit = np.linspace(0, 550, 400)

params, covariance = curve_fit(fit_ex, dis_x_data, CC_Change_y_data, maxfev=5000)

a_fit, b_fit, c_fit = params

y_fit = fit_ex(x_fit, a_fit, b_fit, c_fit)

coefficients = np.polyfit(dis_x_data, CC_Change_y_data, 1)

# Regressionsgerade erstellen
regression_line = np.polyval(coefficients, dis_x_data)

#scatter CC Changeplot
fig, ax = plt.subplots()
plt.scatter(dis_x_data, CC_Change_y_data, label="DeepDenoiser-raw", alpha = 0.6)
plt.plot(dis_x_data, regression_line, color='red', label='Regressionsgerade')
#plt.plot(x_fit, y_fit, label='Fit exp', color='green')
plt.title("CC Change", fontsize=20)
plt.grid(True, color='gray')
plt.axhline(y=0, color='black', linestyle='--')
plt.legend()
plt.xlabel('Distance in km', fontsize=15)
plt.ylabel('Cross-Correlation Coefficient Difference', fontsize=15)
fig.set_size_inches(10, 7, forward=True)
#_median_detrend
flnm= 'CC_Change_REQS_all_V3' + '.pdf'
# fig.savefig(path+flnm, format='pdf')

coefficients_raw = np.polyfit(dis_x_data, CC_y_data, 7)
regression_line_raw = np.polyval(coefficients_raw, x_fit)

coefficients_deno = np.polyfit(dis_x_data, CC_deno_y_data, 7)
regression_line_deno = np.polyval(coefficients_deno, x_fit)

# Kurvenanpassung an die Daten
params, covariance = curve_fit(fit_ex, dis_x_data, CC_y_data, maxfev=5000)
params2, covariance2 = curve_fit(fit_ex, dis_x_data, CC_deno_y_data, maxfev=5000)

# Extrahieren der angepassten Parameter
a_fit, b_fit, c_fit = params
a_fit2, b_fit2, c_fit2 = params2

# Berechnen der y-Werte basierend auf den angepassten Parametern
y_fit = fit_ex(x_fit, a_fit, b_fit, c_fit)
y_fit2 = fit_ex(x_fit, a_fit2, b_fit2, c_fit2)

#scatter CC Changeplot
fig, ax = plt.subplots()
plt.scatter(dis_x_data, CC_y_data, label="Raw", alpha = 0.6)
plt.scatter(dis_x_data, CC_deno_y_data, label="DeepDenoiser", alpha = 0.6)
plt.plot(x_fit, y_fit2, label='Fit DeepDenoiser', color='r')
plt.plot(x_fit, y_fit, label='Fit Raw', color='green')
#plt.plot(x_fit, regression_line_raw, color='green', label='Fit Raw')
#plt.plot(x_fit, regression_line_deno, color='red', label='Fit DeepDenoiser')
plt.title("CC Comparison", fontsize=20)
plt.grid(True, color='gray')
plt.legend()
plt.xlabel('Distance in km', fontsize=15)
plt.ylabel('Cross-Correlation Coefficients', fontsize=15)
fig.set_size_inches(10, 7, forward=True)
#_median_detrend
flnm= 'CC_REQS_all_V3' + '.pdf'
# fig.savefig(path+flnm, format='pdf')

# violinplot plot

# daten gruppieren nach Entfernung 
#%% violinplot 

y_list_100 = []
z_list_100  = []
a_list_100  = []
y_list_200 = []
z_list_200  = []
a_list_200  = []
y_list_300 = []
z_list_300  = []
a_list_300  = []
y_list_400 = []
z_list_400  = []
a_list_400  = []
y_list_500 = []
z_list_500  = []
a_list_500  = []

for x,y,z,a in zip(dis_x_data, CC_Change_y_data, CC_y_data, CC_deno_y_data):
    if x < 100:
        y_list_100.append(y)
        z_list_100.append(z)
        a_list_100.append(a)
        
    elif x < 200:
        y_list_200.append(y)
        z_list_200.append(z)
        a_list_200.append(a)
        
    elif x < 300:
        y_list_300.append(y)
        z_list_300.append(z)
        a_list_300.append(a)
    elif x < 400:
        y_list_400.append(y)
        z_list_400.append(z)
        a_list_400.append(a)
    elif x < 500:
        y_list_500.append(y)
        z_list_500.append(z)
        a_list_500.append(a)
    
pos = [50, 150, 250, 350, 450]
fig, ax = plt.subplots()
plt.violinplot([y_list_100,y_list_200,y_list_300,y_list_400,y_list_500],widths=50, positions=pos, showmeans=True)
plt.title("CC Change in 100 km Steps", fontsize=20)
#plt.legend()
plt.xlabel('Distance in km', fontsize=15)
plt.ylabel('Cross-Correlation Coefficients Difference', fontsize=15)
fig.set_size_inches(10, 7, forward=True)

import matplotlib.patches as mpatches

labels = []
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    
pos = [50, 150, 250, 350, 450]
fig, ax = plt.subplots()
add_label(plt.violinplot([z_list_100,z_list_200,z_list_300,z_list_400,z_list_500], pos,widths=50, showmeans=True), "Raw")
add_label(plt.violinplot([a_list_100,a_list_200,a_list_300,a_list_400,a_list_500], pos,widths=50, showmeans=True), "DeepDenoiser")

plt.title("CC Comparison in 100 km Steps", fontsize=20)
plt.legend(*zip(*labels))
plt.xlabel('Distance in km', fontsize=15)
plt.ylabel('Cross-Correlation Coefficients Difference', fontsize=15)
fig.set_size_inches(10, 7, forward=True)

