# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:29:47 2019

@author: Karra's
"""

"""
 Denoising Autoencoders (dA)
 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials
   
   - Yusuke Sugomori: Stochastic Gradient Descent for Denoising Autoencoders,
   http://yusugomori.com/docs/SGD_DA.pdf
"""


import sys
import numpy 


numpy.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


class dA(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, numpy_rng=None):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
            
        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0

        self.numpy_rng = numpy_rng
        self.x = input
        self.W = W
        self.W_prime = self.W.T
        self.hbias = hbias
        self.vbias = vbias

        # self.params = [self.W, self.hbias, self.vbias]


        
    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.numpy_rng.binomial(size=input.shape,
                                       n=1,
                                       p=1-corruption_level) * input

    # Encode
    def get_hidden_values(self, input):
        return sigmoid(numpy.dot(input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)


    def train(self, lr=0.1, corruption_level=0.3, input=None):
        if input is not None:
            self.x = input

        x = self.x
        tilde_x = self.get_corrupted_input(x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W =  numpy.dot(tilde_x.T, L_h1) + numpy.dot(L_h2.T, y)


        self.W += lr * L_W
        self.hbias += lr * numpy.mean(L_hbias, axis=0)
        self.vbias += lr * numpy.mean(L_vbias, axis=0)



    def negative_log_likelihood(self, corruption_level=0.3):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        

        
        cross_entropy = - numpy.mean(
            numpy.sum(self.x * numpy.log(z) +
            (1 - self.x) * numpy.log(1 - z),
                      axis=1))

        return cross_entropy


    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        return z



def test_dA(data,learning_rate=0.1, corruption_level=0.3, training_epochs=50):
    
    
    rng = numpy.random.RandomState(123)

    # construct dA
    da = dA(input=data, n_visible=data.shape[1], n_hidden=5, numpy_rng=rng)

    # train
    for epoch in range(training_epochs):
        da.train(lr=learning_rate, corruption_level=corruption_level)
        cost = da.negative_log_likelihood(corruption_level=corruption_level)
       # print ( 'Training epoch %d, cost is ' % epoch, cost)
        #learning_rate *= 0.95


    # test
    #x = numpy.array([[1, numpy.nan, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     #                [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]])
    b=da.reconstruct(data)
    return b



df_3=pd.read_csv("D:/ml_github/cat_data_full_data.csv")
df=pd.read_csv("D:/ml_github/cat_data.csv")
mask=df.isnull().astype(int)
df_2=pd.read_csv("D:/ml_github/cat_data_with_null_data.csv")
df_initial_fill=df.fillna(0)
imputed=test_dA(df_initial_fill.values)

imp=pd.DataFrame(imputed,columns=df.columns)

df_imputed=df.fillna(imp)