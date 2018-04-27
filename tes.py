import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv,norm


# prepare dataset from excel
input_data_excel = pd.read_excel("D:/source/dataset.xlsx",header=None)
output_data_excel = pd.read_excel("D:/source/target.xlsx",header=None)
input_data = input_data_excel.values[:,:,np.newaxis]
output_data = output_data_excel.values[:,:,np.newaxis]

#Qdot max = 1.2e+3
#E max = 12 V

class Network(object):
    def __init__(self, sizes , input_data , output_data ):

        # Separate dataset to training and validation data
        self.in_train = input_data[0:15001]
        self.out_train = output_data[0:15001]
        self.in_test = input_data[15001:20000]
        self.out_test = output_data[15001:20000]

        self.num_layers = len(sizes)
        self.sizes = sizes

        batas = 0.7 * (self.sizes[1])**(1/self.sizes[0])
        self.biases =  [ batas - (2*batas)*np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [ 0.5 - 1*np.random.rand(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # nguyen weight inisialitation
        for i  in range (self.sizes[1]):
            new_norm =  batas / norm(self.weights[0][i])
            self.weights[0][i] = self.weights[0][i] * new_norm


    def levenberg_marquardt(self, mu, vartetha, iter_max , target_error):


        iter = 0 ;
        sum_error = np.zeros(iter_max+1)
        sum_error[iter] = target_error + 1
        sum_error_test = np.zeros(iter_max+1)

        while ((sum_error[iter] > target_error ) and (iter < iter_max) ):

            J, v, l = self.jacobian_calculation(self.in_train,self.out_train)
            sum_error[iter] = sum(v*v)
            v_test = self.feedforward(self.weights,self.biases,self.in_test,self.out_test)
            sum_error_test[iter] = sum(v_test*v_test)

            tes_weights,tes_biases = self.updateparameter(J,mu,l,v)
            new_v = self.feedforward(tes_weights,tes_biases,self.in_train,self.out_train)
            sum_error[iter+1] = sum(new_v*new_v)

            if (sum_error[iter+1] < sum_error[iter]):
                mu  = mu / vartetha
            else:
                count = 0
                while ((sum_error[iter+1] >= sum_error[iter]  ) and ( count < 5 )):
                    mu = mu * vartetha
                    tes_weights,test_biases = self.updateparameter(J,mu,l,v)
                    new_v = self.feedforward(tes_weights,tes_biases,self.in_train,self.out_train)
                    sum_error[iter+1] = sum(new_v*new_v)
                    count += 1


            self.weights = tes_weights
            self.biases = tes_biases
            iter += 1

        v_test = self.feedforward(self.weights,self.biases,self.in_test,self.out_test)
        sum_error_test[iter] = sum(v_test*v_test)
        plt.plot(sum_error[0:iter+1])
        plt.show()


    def updateparameter(self, J, mu ,l, v):
        # calculate delta weights and biases
        G = inv(np.dot(J.T,J) + mu*np.eye(l))
        delta_parameter = np.dot(np.dot(G,J.T),v)

        # Update weights and biases
        tes_weights = self.weights
        tes_biases = self.biases

        l =0
        for m in range(0, self.num_layers-1):
            for i in range(self.sizes[m+1]+1):
                if i < self.sizes[m+1] :
                    tes_weights[m][i] = tes_weights[m][i] - delta_parameter[l:l+self.sizes[m]].T
                    l += self.sizes[m]
                else :
                    tes_biases[m] = tes_biases[m] - delta_parameter[l:l+self.sizes[m+1]]
                    l += self.sizes[m+1]

        return(tes_weights,tes_biases)


    def feedforward(self, w, b, x, y):
        num_dataset = len(x)
        v = np.zeros((num_dataset * self.sizes[-1],1))

        #calculation for all data in dataset
        for training in range(num_dataset):
            h = training * self.sizes[-1]
            a = x[training]

            #feedforward
            for bias, weight in zip(b[0:-1], w[0:-1]):
                z = np.dot(weight,a) + bias
                a = tansig(z)


            z = np.dot(w[-1],a) + b[-1]
            a = z

            v[h:h+self.sizes[-1]] = y[training] - a

        return(v)

    def jacobian_calculation(self, x, y):

        num_dataset = len(x)

        # arange jacobian matrices
        J_row = num_dataset * self.sizes[-1]
        J_col = 0
        for i in range(1,len(self.sizes)):
            J_col += self.sizes[i] * (self.sizes[i-1]+1)

        J = np.zeros((J_row,J_col))
        v = np.zeros((J_row,1))

        # calculation for all data in dataset
        for training in range(num_dataset):
            h = training * self.sizes[-1]
            activation = x[training]
            activations = [activation] # list to store all the activations, layer by layer

            #feedforward
            for b, w in zip(self.biases[0:-1], self.weights[0:-1]):
                z = np.dot(w,activation) + b
                activation = tansig(z)
                activations.append(activation)

            z = np.dot(self.weights[-1],activation) + self.biases[-1]
            activation = z
            activations.append(activation)

            v[h:h+self.sizes[-1]] = y[training] - activation

            # Sensitivity calculation
            S_layer = -1 * np.eye(self.sizes[-1])
            S = [S_layer]

            for l in range(2, self.num_layers):
                S_layer = np.dot(np.dot(np.diag(dtansig(activations[-l].T)[0]),net.weights[-l+1].T),S_layer)
                S.append(S_layer)

            # Jacobian calculation
            l = 0
            for m in range(0, self.num_layers-1):
                for i in range(self.sizes[m+1]+1):
                    if i < self.sizes[m+1] :
                        J[h:h+self.sizes[-1],l:l+self.sizes[m]] = np.dot(S[-m-1][i,:,np.newaxis],activations[m].T)
                        l += self.sizes[m]
                    else :
                        J[h:h+self.sizes[-1],l:l+self.sizes[m+1]] = S[-m-1].T
                        l += self.sizes[m+1]

        return(J,v,J_col)


# tansigmoid function
def tansig(x):
    return ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))

def dtansig(x):
    return (1-x*x)


#main
net = Network([6,20,3],input_data,output_data)
mu = 0.01
vartetha = 5
iter_max = 20
target_error = 0.01
net.levenberg_marquardt( mu, vartetha, iter_max , target_error)
