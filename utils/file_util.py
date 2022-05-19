import pickle
import pandas as pd

def save(variable, filename):
    c_file = open(filename,'wb')
    pickle.dump(variable, c_file,protocol = 4)
    c_file.close()
    
def load(filename):
    d_file = open(filename,'rb+')
    data = pickle.load(d_file)
    d_file.close()
    return data

