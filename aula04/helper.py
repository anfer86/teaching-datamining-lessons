import numpy as np

def regression_to_equation (columns, coefs, intercept, nround = 3):
    coefs = np.round(coefs, nround).astype(str)    
    intercept = np.round(intercept, nround).astype(str)    
    str1 = " + ".join( [(coef + ' x '+ column) for column, coef in zip(columns, coefs)] )        
    return 'Y = ' + str1 + ' + ' + intercept;