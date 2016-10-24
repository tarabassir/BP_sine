#!/usr/bin/python

import math
import random

# Neural Network class
class nn:
#------------------------------------------------------------------------------
    def __init__(self, weights = {}):
        self.__weights = weights
        self.__total_error = 0

        if len(self.__weights) == 0:
            w = self.__weights
            w.update({'wi1h1' : random.random()})
            w.update({'wi2h1' : random.random()})
            w.update({'wi3h1' : random.random()})
            w.update({'wi4h1' : random.random()})
            w.update({'wi1h2' : random.random()})
            w.update({'wi2h2' : random.random()})
            w.update({'wi3h2' : random.random()})
            w.update({'wi4h2' : random.random()})
            w.update({'wi1h3' : random.random()})
            w.update({'wi2h3' : random.random()})
            w.update({'wi3h3' : random.random()})
            w.update({'wi4h3' : random.random()})
            w.update({'wi1h4' : random.random()})
            w.update({'wi2h4' : random.random()})
            w.update({'wi3h4' : random.random()})
            w.update({'wi4h4' : random.random()})
            w.update({'wh1o' : random.random()})
            w.update({'wh2o' : random.random()})
            w.update({'wh3o' : random.random()})
            w.update({'wh4o' : random.random()})

        self.__neurons = {'i1' : 0, 'i2' : 0, 'i3' : 0, 'i4' : 0,
                          'h1net' : 0, 'h2net' : 0, 'h3net' : 0, 'h4net': 0,
                          'h1out' : 0, 'h2out' : 0, 'h3out' : 0, 'h4out' : 0,
                          'oanet' : 0, 'oaout' : 0, 'ot' : 0}

        # Partial derivatives
        self.__pd = {'et_wi1h1' : 0, 'et_wi2h1' : 0, 'et_wi3h1' : 0, 'et_wi4h1' : 0,
            'et_wi1h2' : 0, 'et_wi2h2' : 0, 'et_wi3h2' : 0, 'et_wi4h2' : 0,
            'et_wi1h3' : 0, 'et_wi2h3' : 0, 'et_wi3h3' : 0, 'et_wi4h3' : 0,
            'et_wi1h4' : 0, 'et_wi2h4' : 0, 'et_wi3h4' : 0, 'et_wi4h4' : 0,
            'et_wh1o' : 0, 'et_wh2o' : 0, 'et_wh3o' : 0, 'et_wh4o' : 0}
#------------------------------------------------------------------------------
    def display_weights(self):
        for __key in self.__weights.keys():
            print(repr(__key).rjust(10) + ": " + repr(self.__weights[__key]))
#------------------------------------------------------------------------------
    def display_neurons(self):
        for __key in self.__neurons.keys():
            print(repr(__key).rjust(10) + ": " + repr(self.__neurons[__key]))
#------------------------------------------------------------------------------
    def display_total_error(self):
        print("Total Error: ", self.__total_error)
#------------------------------------------------------------------------------
    def display_pd(self):
        for __key in self.__pd.keys():
            print(repr(__key).rjust(10) + ": " + repr(self.__pd[__key]))
#------------------------------------------------------------------------------
# Forward pass from the inputs to the hidden layer nodes
    def __fp_to_hidden(self):
        nnn = self.__neurons
        w = self.__weights

        nnn['h1net'] = (nnn['i1'] * w['wi1h1']) + (nnn['i2'] * w['wi2h1']) + \
                (nnn['i3'] * w['wi3h1']) + (nnn['i4'] * w['wi4h1'])
        nnn['h2net'] = (nnn['i1'] * w['wi1h2']) + (nnn['i2'] * w['wi2h2']) + \
                (nnn['i3'] * w['wi3h2']) + (nnn['i4'] * w['wi4h2'])
        nnn['h3net'] = (nnn['i1'] * w['wi1h3']) + (nnn['i2'] * w['wi2h3']) + \
                (nnn['i3'] * w['wi3h3']) + (nnn['i4'] * w['wi4h3'])
        nnn['h4net'] = (nnn['i1'] * w['wi1h4']) + (nnn['i2'] * w['wi2h4']) + \
                (nnn['i3'] * w['wi3h4']) + (nnn['i4'] * w['wi4h4'])

        nnn['h1out'] = 1 / (1 + math.exp(-1 * nnn['h1net']))
        nnn['h2out'] = 1 / (1 + math.exp(-1 * nnn['h2net']))
        nnn['h3out'] = 1 / (1 + math.exp(-1 * nnn['h3net']))
        nnn['h4out'] = 1 / (1 + math.exp(-1 * nnn['h4net']))
#------------------------------------------------------------------------------
# Forward pass from the hidden layer nodes to the output
    def __fp_to_output(self):
        nnn = self.__neurons
        w = self.__weights

        nnn['oanet'] = (nnn['h1out'] * w['wh1o']) + \
                (nnn['h2out'] * w['wh2o']) + \
                (nnn['h3out'] * w['wh3o']) + \
                (nnn['h4out'] * w['wh4o'])

        nnn['oaout'] = 1 / (1 + math.exp(-1 * nnn['oanet']))
#------------------------------------------------------------------------------
# Calculates total error
    def __calc_error(self):
        nnn = self.__neurons

        self.__total_error = (1 / 2) * ((nnn['ot'] - nnn['oaout']) ** 2)
#------------------------------------------------------------------------------
# Backward pass from output to hidden layer nodes
    def __bp_to_hidden(self):
        nnn = self.__neurons
        pd = self.__pd

        pd['et_wh1o'] = -1 * (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * nnn['h1out']
        pd['et_wh2o'] = -1 * (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * nnn['h2out']
        pd['et_wh3o'] = -1 * (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * nnn['h3out']
        pd['et_wh4o'] = -1 * (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * nnn['h4out']
#------------------------------------------------------------------------------
# Backward pass from hidden layer nodes to inputs
    def __bp_to_inputs(self):
        nnn = self.__neurons
        w = self.__weights
        pd = self.__pd

        pd['et_wi1h1'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh1o'] * (nnn['h1out'] * (1 - nnn['h1out'])) * nnn['i1']
        pd['et_wi2h1'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh1o'] * (nnn['h1out'] * (1 - nnn['h1out'])) * nnn['i2']
        pd['et_wi3h1'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh1o'] * (nnn['h1out'] * (1 - nnn['h1out'])) * nnn['i3']
        pd['et_wi4h1'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh1o'] * (nnn['h1out'] * (1 - nnn['h1out'])) * nnn['i4']
        pd['et_wi1h2'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh2o'] * (nnn['h2out'] * (1 - nnn['h2out'])) * nnn['i1']
        pd['et_wi2h2'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh2o'] * (nnn['h2out'] * (1 - nnn['h2out'])) * nnn['i2']
        pd['et_wi3h2'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh2o'] * (nnn['h2out'] * (1 - nnn['h2out'])) * nnn['i3']
        pd['et_wi4h2'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh2o'] * (nnn['h2out'] * (1 - nnn['h2out'])) * nnn['i4']
        pd['et_wi1h3'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh3o'] * (nnn['h3out'] * (1 - nnn['h3out'])) * nnn['i1']
        pd['et_wi2h3'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh3o'] * (nnn['h3out'] * (1 - nnn['h3out'])) * nnn['i2']
        pd['et_wi3h3'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh3o'] * (nnn['h3out'] * (1 - nnn['h3out'])) * nnn['i3']
        pd['et_wi4h3'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh3o'] * (nnn['h3out'] * (1 - nnn['h3out'])) * nnn['i4']
        pd['et_wi1h4'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh4o'] * (nnn['h4out'] * (1 - nnn['h4out'])) * nnn['i1']
        pd['et_wi2h4'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh4o'] * (nnn['h4out'] * (1 - nnn['h4out'])) * nnn['i2']
        pd['et_wi3h4'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh4o'] * (nnn['h4out'] * (1 - nnn['h4out'])) * nnn['i3']
        pd['et_wi4h4'] = (nnn['ot'] - nnn['oaout']) * \
                (nnn['oaout'] * (1 - nnn['oaout'])) * \
                w['wh4o'] * (nnn['h4out'] * (1 - nnn['h4out'])) * nnn['i4']
#------------------------------------------------------------------------------
# Return total error
    def get_total_error(self):
        return self.__total_error
#------------------------------------------------------------------------------
    def get_predicted_output(self):
        return self.__neurons['oaout']
#------------------------------------------------------------------------------
# Update the weights of the neural network
    def __update_weights(self):
        w = self.__weights
        pd = self.__pd

        w['wi1h1'] = w['wi1h1'] - pd['et_wi1h1']
        w['wi2h1'] = w['wi2h1'] - pd['et_wi2h1']
        w['wi3h1'] = w['wi3h1'] - pd['et_wi3h1']
        w['wi4h1'] = w['wi4h1'] - pd['et_wi4h1']

        w['wi1h2'] = w['wi1h2'] - pd['et_wi1h2']
        w['wi2h2'] = w['wi2h2'] - pd['et_wi2h2']
        w['wi3h2'] = w['wi3h2'] - pd['et_wi3h2']
        w['wi4h2'] = w['wi4h2'] - pd['et_wi4h2']

        w['wi1h3'] = w['wi1h3'] - pd['et_wi1h3']
        w['wi2h3'] = w['wi2h3'] - pd['et_wi2h3']
        w['wi3h3'] = w['wi3h3'] - pd['et_wi3h3']
        w['wi4h3'] = w['wi4h3'] - pd['et_wi4h3']

        w['wi1h4'] = w['wi1h4'] - pd['et_wi1h4']
        w['wi2h4'] = w['wi2h4'] - pd['et_wi2h4']
        w['wi3h4'] = w['wi3h4'] - pd['et_wi3h4']
        w['wi4h4'] = w['wi4h4'] - pd['et_wi4h4']

        w['wh1o'] = w['wh1o'] - pd['et_wh1o']
        w['wh2o'] = w['wh2o'] - pd['et_wh2o']
        w['wh3o'] = w['wh3o'] - pd['et_wh3o']
        w['wh4o'] = w['wh4o'] - pd['et_wh4o']
#------------------------------------------------------------------------------
# The forefront of the backpropagation algorithm
    def bp(self, dataset):
        nnn = self.__neurons

        nnn['i1'] = dataset['i1']
        nnn['i2'] = dataset['i2']
        nnn['i3'] = dataset['i3']
        nnn['i4'] = dataset['i4']
        nnn['ot'] = dataset['ot']

        self.__fp_to_hidden()
        self.__fp_to_output()
        self.__calc_error()
        self.__bp_to_hidden()
        self.__bp_to_inputs()
        self.__update_weights()
#------------------------------------------------------------------------------

inputs = [0, 0.2692, 0.5017, 0.6938, 0.8415,
	  0.9408, 0.9878, 0.9786, 0.9093, 0.7789,
	  0.5982, 0.3810, 0.1411, -0.1075, -0.3501,
	  -0.5715,   -0.7568,   -0.8928,   -0.9742,   -0.9974,
	  -0.9589,   -0.8579,   -0.7032,   -0.5066,   -0.2794,
	  -0.0340,    0.2144,    0.4501,    0.6570,    0.8210,
	  0.9349,    0.9929,    0.9894,    0.9213,    0.7958,
	  0.6229,    0.4121,    0.1745,   -0.0749,   -0.3198,
	  -0.5440,   -0.7327,   -0.8768,   -0.9684,   -1.0000,
	  -0.9663,   -0.8726,   -0.7267,   -0.5366,   -0.3114,
	  -0.0661,    0.1832,    0.4202,    0.6297,    0.8011,
	  0.9246,    0.9906,    0.9919,    0.9318,    0.8160,
	  0.6503,    0.4421,    0.2058,   -0.0429,   -0.2879,
	  -0.5141,   -0.7094,   -0.8623,   -0.9614,   -0.9977,
	  -0.9724,   -0.8890,   -0.7510,   -0.5640,   -0.3414,
 	  -0.0983,    0.1499,    0.3883,    0.6036,    0.7828,
	  0.9129,    0.9835,    0.9934,    0.9440,    0.8367,
	  0.6748,    0.4706,    0.2380,   -0.0089,   -0.2556,
	  -0.4874,   -0.6893,   -0.8462,   -0.9460,   -0.9879,
	  -0.9737,   -0.9056,   -0.7854,   -0.6151,   -0.3968]
	  
	  
	  


if __name__ == "__main__":
	nn_object = nn()
	nn_object.display_total_error()
    
	i = 0
	epoch = 0
	predicted_vals = []
	
	nn_object.bp({'i1' : inputs[i], 'i2' :  inputs[i + 1], 'i3' : inputs[i + 2], 'i4' : inputs[i + 3], 'ot' : inputs[i + 4]})

	nn_object.display_total_error()
	
	while nn_object.get_total_error() > (10 ** -10):
		if (len(predicted_vals) < 100):
			predicted_vals.insert(0, nn_object.get_predicted_output())
		else:
			predicted_vals.pop()
			predicted_vals.insert(0, nn_object.get_predicted_output())
		i = i % 96
		nn_object.bp({'i1' : inputs[i], 'i2' :  inputs[i + 1], 'i3' : inputs[i + 2], 'i4' : inputs[i + 3], 'ot' : inputs[i + 4]})

		
		i += 1
		epoch += 1
		

	nn_object.display_neurons()	
	nn_object.display_total_error()
	print("Epoch: ", epoch)
	print(predicted_vals)      
	
		
		

