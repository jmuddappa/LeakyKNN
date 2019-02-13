import pandas as pd
import numpy as np
import math as m

#function to calculate distance using haversine distance formula between house i and house j
def distance(origin, destination): 
    lat1, lon1 = origin[0], origin[1]
    lat2, lon2 = destination[0], destination[1]
    radius = 6371 # km

    dlat = m.radians(lat2-lat1)
    dlon = m.radians(lon2-lon1)
    a = m.sin(dlat/2) * m.sin(dlat/2) + m.cos(m.radians(lat1)) \
       * m.cos(m.radians(lat2)) * m.sin(dlon/2) * m.sin(dlon/2)
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1-a))
    d = radius * c
    return d

#function to obtain k neighbors
def get_k_neighbors(neighbors,d,k):    
    kneighbors = [None]*(k)
    for i in range(1,k+1): 
        kneighbors[i-1] = neighbors[d[i]]
    return kneighbors

#function to get weights for weighted KNN algorithm where w = inverse of distance from house i
def get_weights(kneighbors):
    w = [None]*(len(kneighbors))
    kneighbors[kneighbors[:,0]==0,0]=0.0000001 #change 0 values to a very small value to prevent divide by 0 errors
    w = 1/kneighbors[:,0]
    w = w/sum(w)
    return w

#function to perform KNN where inputs are 
#k - the number of neighbors for each house, 
#file - the file name to be processed, 
#N - is max number of rows to process with time constraint, if no value is given then process entire dataset
def leakyKNN(k,file,N=-1):
    housing = pd.read_csv(file) 
    data = housing.values #converted pandas file to numpy array
    data = data[data[:,3]>0] 
    if(N==-1):
        N=len(data)
    MRAE=[]
    for curr_house_index in range(N):
        #if(curr_house_index%1000==0):
        print("House number:", curr_house_index)
        valid_neighbors = data[data[curr_house_index, 2]>data[:,2]] #get valid neighbors with time leak prevented
        distance_from_origin = [None]*len(valid_neighbors)
 
        for i in range(len(valid_neighbors)):
           distance_from_origin[i] = distance(data[curr_house_index],valid_neighbors[i])
 
        valid_neighbors[:,0] = distance_from_origin #store distances in first column
        distance_from_origin=sorted(range(len(distance_from_origin)), key=distance_from_origin.__getitem__) #get valid indexes
        if(len(distance_from_origin)<=k): #case where less than k valid neighbors exist 
            kneighbors = [None]*(len(distance_from_origin)+1)
            kneighbors=valid_neighbors 
        else:
            #kneighbors = [None]*(k)
            kneighbors=get_k_neighbors(valid_neighbors,distance_from_origin,k)
        kneighbors=np.delete(kneighbors, [1,2], axis=1) #k nearest neighbors with distance and value
        w=get_weights(kneighbors)
        y_pred = sum(w*kneighbors[:,1]) #predicted value
        y = data[curr_house_index, 3] #actual value of house
        MRAE.append(np.abs(y_pred - y) / y) 
    print(np.median(MRAE)) #report median relative absolute error
    return MRAE

if __name__== "__main__":
    MRAE=leakyKNN(4,"data.csv",100)

