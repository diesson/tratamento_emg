import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import statistics as st
import matplotlib.pyplot as plt
import math
import os
from sklearn import tree
from sklearn.tree import export_graphviz

def lista_arquivos(diretorio='./', tipo=".txt"):
    caminhos = [os.path.join(nome, diretorio) for nome in os.listdir(diretorio)]
    
    arquivos = []
    for caminho, _, arquivo in os.walk(diretorio):
        for arq in arquivo:
            arquivos.append(caminho+'/'+arq)
    
    lista = [arq for arq in arquivos if arq.lower().endswith(tipo)]
    #lista = ['./dados/'+lista[n] for n in range(0,len(lista))]
    
    return lista

def abrir_arquivos(arq, qtd):
    n = 0
    dados = []
    while n < qtd:
        val = np.genfromtxt(arq[n], delimiter=',')
        dados.append(val)
        n+=1
    return dados

def energia_sinal(sinal, block_size = 64):
    N1 = len(sinal)-1
    n = 0
    j = 0
    energia = []
    while(n < N1):
        i = 0
        val = 0
        while(i < 2*block_size):
            if((n+i) >= N1):
                break

            val += sinal[n+i]**2
            i+=1
        while(j < n):
            energia.append(val/i)
            j+=1

        n+=block_size
    while(j <= N1):
        energia.append(val/i)
        j+=1

    energia = np.array(energia)
    
    return energia

def detect_ativ(sinal, th, block_size = 64):
    
    tr = th*np.mean(sinal)
    ativacao = np.zeros(len(sinal))
    ativ_i = [0]
    j = block_size
    i = 0
    val = 0
    flag_ativ = 0
    
    while(j < (len(sinal)-1)):
        while((i < j) and (i < (len(sinal)-1))):
            if( (j+2*block_size) < (len(sinal)-1) ):
                if((sinal[j] > tr) and (sinal[j+block_size] > tr) and (sinal[j+2*block_size] > tr)):
                    if((sinal[j-block_size] < tr) and (sinal[j-2*block_size] < tr)):
                        val = 1
                        if(flag_ativ == 0):
                            ativ_i.append(i)
                            flag_ativ = 1
                        
            if( (j+2*block_size) < (len(sinal)-1) ):
                if((sinal[j] > tr) and (sinal[j-block_size] > tr) and (sinal[j-2*block_size] > tr)):
                    if((sinal[j+block_size] < tr) and (sinal[j+2*block_size] < tr)):
                        val = 0
                        if(flag_ativ == 1):
                            ativ_i.append(i)
                            flag_ativ = 0
                        
            ativacao[i] = val
            i += 1
        j+=block_size
        
    ativ_i.append(len(sinal)-1)
    return ativacao, ativ_i, tr

def fft_sinal(sinal, fs):
    fft_sinal = np.fft.fft(sinal)
    freq_sinal = np.fft.fftfreq(sinal.size, 1/fs)
    
    return freq_sinal, fft_sinal

def separa_sinal(temp, sinal, pontos):
    
    bloco_x = []
    bloco_y = []
    qtd_blocos = (len(pontos)-1)
    i = 0
    while(i < qtd_blocos):
        bloco_y.append(sinal[pontos[i]:pontos[i+1]])
        bloco_x.append(temp[pontos[i]:pontos[i+1]])
        i+=1
    
    return [bloco_x, bloco_y]

def emg_media(sinal, block_size = 64):
    
    media = np.zeros(len(sinal))
    j = block_size
    i = 0
    
    while(j < (len(sinal)-1)):
        while((i < j) and (i < (len(sinal)-1))):
            if( (j+2*block_size) < (len(sinal)-1) ):
                media[i] = np.mean(np.abs(sinal[j:j+block_size]))
            i += 1
        j+=block_size
        
    return media

def emg_rms(sinal, block_size = 64):
    
    rms = np.zeros(len(sinal))
    j = block_size
    i = 0
    
    while(j < (len(sinal)-1)):
        while((i < j) and (i < (len(sinal)-1))):
            if( (j+2*block_size) < (len(sinal)-1) ):
                rms[i] = np.sqrt(np.mean(sinal[j:j+block_size]**2))
            i += 1
        j+=block_size
        
    return rms

def emg_variancia(sinal, block_size = 64):
    
    media = np.zeros(len(sinal))
    j = block_size
    i = 0
    flag_ativ = 0
    
    while(j < (len(sinal)-1)):
        while((i < j) and (i < (len(sinal)-1))):
            if( (j+2*block_size) < (len(sinal)-1) ):
                media[i] = np.var(np.abs(sinal[j:j+block_size]))
            i += 1
        j+=block_size
        
    return media

def emg_waveform_length(sinal):
    
    wl = np.zeros(len(sinal))
    i = 0
    
    while(i < (len(sinal)-1)):
        wl[i] = sinal[i+1] - sinal[i]
        i += 1
    wl[i] = 0
    
    return wl

def ar_pred(train, test):
    model = AR(train).fit()
    window = model.k_ar
    coef = model.params

    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]

    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
    
    return predictions

def auto_reg(sinal, n):
    
    lst_n = len(sinal)
    x = np.zeros((n, lst_n))
    i = 1
    while(i <= n):
        x[i-1, i:lst_n] = sinal[0:lst_n-i]
        i += 1
    
    vet_a = np.zeros((lst_n, n))
    val_var = np.zeros(lst_n)
    i = 0
    y = []
    while(i < n):
        val_x = x[0:i+1,:]
        matriz_x = np.matmul(val_x, val_x.transpose())
        inv_x = np.linalg.inv(matriz_x)
        
        if( i > 0):
            vet_a[i, 0:i+1] = np.matmul(sinal, np.matmul(val_x.transpose(), inv_x))
        else:
            vet_a[i, 0] = np.matmul(sinal, inv_x * val_x.transpose())
        
        val_var[i] = np.var( sinal - np.matmul(vet_a[i, 0:i+1], x[0:i+1, :]))
        mdl = lst_n*np.log10(val_var[i]) + i*np.log10(lst_n)
        y.append(mdl)
        
        i += 1

    sinal_criado = np.matmul(vet_a[np.argmin(y), 0:np.argmin(y)+1], x[0:np.argmin(y)+1, :])
    
    return np.argmin(y), vet_a[np.argmin(y), 0:np.argmin(y)], y, val_var[np.argmin(y)], sinal_criado