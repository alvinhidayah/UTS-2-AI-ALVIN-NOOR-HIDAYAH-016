#ALVIN NOOR HIDAYAH /21091397016/ B
import numpy as np
#inisialisasi variabel
inputs = [[2.0, 1.0, 4.0, 2.2, 1.0, 0.6, 0.2, 0.1, 0.12, 0.2],
		 [1.0, 3.0,-1.0, 1.0, 2.0, 0.4, 0.1, 0.3, 0.4, 0.12],
		 [-1.2, 2.2, 3.4,-0.7, 4.2, 5.2, 0.2, 0.3, 0.4, 0.13],
         [1.2, 0.8, 0.3, 0.4, 0.6, 4.5, 2.3, 4.5, 5.1, 0.2],
         [3.1, 3.2, 3.3, 4.5, 6.1, 6.6, 1.2, 0.2, -0.2, 0.2],
         [1.2, 3.2, 3.4,-1.7, 4.2, 3.2, 0.2, 0.1, 0.3, 2.1],]
#banyak bobot neuron
weights = [[1.3, 1.2, 3.4,-0.7, 9.2, 0.4, 1.2, 0.2, 0.2, 0.1],
          [1.2, 2.9, 7.4,0.5, 9.2, 1.2, 0.2, 0.3, 9.0, 1.3],
          [8.2, 8.2, 3.4,-1.7, 8.2, 9.3, 0.2, 0.1, 0.2, 0.2],
          [7.2, 5.7, 3.4,0.4, 2.2, 9.9, 0.2, 3.1, 3.3, 0.1],
          [1.5, 2.9, 3.4,-0.1, 7.2, 9.3, 0.1, 9.2, -0.3, 0.2]]
#banyak bias weights 1 sebanyak 5 input
biases= [0.1, 0.3, 0.4, 0.2, 0.1]
#bobot neuron kedua
weights2= [[1.2, 1.2, 3.4,-0.7, 1.2, ],
          [1.2, 2.13, 7.4, 0.6, 0.2, ],
          [-0.2, 0.2, 3.4,-1.7, 8.2, ]]  
 #bias di weight 2 sebanyak 3 input           
biases2= [-1, 2, -0.5,]

#perhitungan hidden layer 
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
#perhitungan hidden layer 2 setelah dikaliakan weight 1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)