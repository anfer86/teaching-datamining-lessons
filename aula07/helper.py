import numpy as np
import pandas as pd

def entropia (vp):
    somatorio = 0;
    for p in vp :
        somatorio += 0 if (p == 0) else p * np.log2(p)    
    return somatorio * (-1)

def entropia_valores(valores):
    import collections
    counter = collections.Counter(valores)
    vp = [ (n/len(valores)) for n in counter.values() ]    
    return entropia(vp)    

def entropia_atributo(X, atributo, classe) :
    somatorio = 0    
    for valor, Xj in X.groupby(atributo):        
        p_Xj = len(Xj) / len(X)        
        E_Xj = entropia_valores(Xj[classe])
        somatorio += p_Xj * E_Xj        
    return somatorio

def entropia_atributo_detalhes(X, atributo, classe) :
    somatorio = 0
    detalhes = pd.DataFrame(columns = ['valor','prop_Xj'])
    for valor, Xj in X.groupby(atributo):        
        p_Xj = len(Xj) / len(X)        
        E_Xj = entropia_valores(Xj[classe])
        somatorio += p_Xj * E_Xj
        detalhes = detalhes.append({'valor' : valor, 'prop_Xj' : p_Xj, 'E(Xj)' : E_Xj }, 1)
    return somatorio, detalhes

def ganho_de_informacao(X,atributo,classe) :
    E_X = entropia_valores(X[classe])
    E_X_A = entropia_atributo(X,atributo,classe)
    return  E_X - E_X_A

def razao_de_ganho (X, atributo, classe):
    IG = ganho_de_informacao(X, atributo, classe)
    I = entropia_valores(X[atributo])
    return IG / I

def ganho_de_informacao_todos(df,classe):
    return pd.DataFrame({ 
                'InformationGain': df.drop([classe],1).apply(
                lambda x : ganho_de_informacao(df,x.name,classe),0),
    })
