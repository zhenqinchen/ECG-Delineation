import numpy as np
from .Config import Config


def __get_points__(signal, type = 'qrs'):
    
    L = len(signal)
    starts = []
    ends = []
    start = -1
    end = -1
    TOL = 40
    ignore_L = 5

    cnt = 0
    for i in range(L):
        if(signal[i] == 1):
            if cnt == 0:
                start = i
                starts.append(start)
            
            cnt+=1;
            if i == L - 1:
                end = i
                ends.append(end)
        elif cnt > 0:
            end = i - 1
            ends.append(end)
            cnt = 0
            
    

    starts_new = []
    ends_new = []
    i = 0
    while(i < len(starts)):
       
        end = ends[i]
        starts_new.append(starts[i])    
        while(i < len(starts) - 1 and starts[i+1] - end < ignore_L): 
            end = ends[i+1]
            i+=1
        ends_new.append(end)
        i+=1
        

    
   
    max_L = 5
    start = -1
    end = -1
    for i in range(len(starts_new)):

        if type == 'p':
            if starts_new[i]>= L/2 :
                continue
        elif type == 't':
            if ends_new[i]<= L/2 :  
                continue
        elif type == 'qrs':
            r_ = (ends_new[i] + starts_new[i])/2
            if r_ > L/2 + TOL or r_ < L/2 - TOL:
                continue

        
        l = ends_new[i] - starts_new[i]
        if max_L < l:
            max_L = l
            start = starts_new[i]
            end = ends_new[i]
    return start,end


def get_predict_points(labels): 
    points = np.zeros((labels.shape[0], 6))
    L = labels.shape[1]
    FC = 250
    for i in range(labels.shape[0]):
        p_label = np.zeros((L))
        qrs_label = np.zeros((L))
        t_label = np.zeros((L))
        for j in range(L):
            if labels[i, j] == Config.P_H:
                p_label[j] = 1
            elif labels[i, j] == Config.QRS_H:
                qrs_label[j] = 1
            elif labels[i, j] == Config.T_H:
                t_label[j] = 1

        q_start,q_end = __get_points__(qrs_label, type='qrs')
        p_start,p_end = __get_points__(p_label, type='p')
        t_start,t_end = __get_points__(t_label, type='t')

        point= [p_start,p_end, q_start,q_end, t_start,t_end]
        points[i,:] = point
    return points

