import numpy as np
from numpy import genfromtxt
import numpy.linalg as la
import math as math

sim = np.zeros([50, 50])
nhood = np.zeros([50, 5], dtype=int)
np.set_printoptions(precision=3, suppress=True, linewidth=1000)

def load_csv(filename):
    my_data = genfromtxt(filename, delimiter=',')
    my_data = my_data[:, 2:]
    return my_data

def similarity(item1, item2, index1, index2):
    dotProd = np.dot(item1, item2)
    normProd = la.norm(item1) * la.norm(item2)
    cosTheta = min(dotProd / normProd, 1)
    angle = math.acos(cosTheta)
    angle = math.degrees(angle)
    sim[index1, index2] = angle


def neighborhood(similarity):
    for i in range(50):
        nhood[i, :] = np.argpartition(similarity[i, :], -5)[:5]

def fillInValues(my_data, nhood):
    for i in range(50):
        dataZero = np.flatnonzero(my_data[i,:] == 0)
        for j in range(len(dataZero)):
            my_data[i,dataZero[j]] = weightedAverage(dataZero[j], nhood[i,:], my_data, i)

def columnMean(inCol):
    sum = 0
    avg = 0
    ctr = 0
    for i in range(50):
        sum += inCol[i]
        if inCol[i] is not 0:
           ctr += 1
    avg = sum/ctr
    return avg

def weightedAverage(zeroIndices, nhood, my_data, i):
    denom = 0
    numerator = 0
    nonZeroInd = np.where(my_data[nhood[:],zeroIndices] != 0.0)
    for k in range(5):
        denom += sim[i, nhood[k]]
        if my_data[nhood[k], zeroIndices] != 0.0:
            numerator += sim[nhood[k],i]*my_data[nhood[k],zeroIndices]
        else:
            numerator += sim[nhood[k],i]*columnMean(my_data[:, zeroIndices])
    return numerator/denom

def main():
    data = np.random.randint(6, size=(50, 25))

    print("data original ground truth")
    for i in range(len(data)):
        print(data[i,:])

    np.around(data, decimals=2)
    np.savetxt("../data/before_filling.csv", data, delimiter=",", fmt ='%f')

    for i in range(50):
        for j in range(50):
            similarity(data[i, :], data[j, :], i, j)

    neighborhood(sim)

    fillInValues(data, nhood)

    print('similarity')
    for i in range(50):
        print(sim[i,:])
    np.savetxt("../data/similarity_matrix.csv", sim, delimiter=",", fmt ='%f')

    print('neighborhood')
    for i in range(50):
        print(nhood[i,:])
    np.savetxt("../data/neighborhood_matrix.csv", nhood, delimiter=",", fmt ='%f')

    np.savetxt("../data/filled_ratings.csv", data, delimiter=",", fmt = '%f')


if __name__ == "__main__":
    main()
