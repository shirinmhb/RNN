from __future__ import annotations
import numpy as np
import time
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.recurrent import ElmanSimpleRecurrent 
from pyneurgen.recurrent import NARXRecurrent
import datetime

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class Rnn():
  def __init__(self, w) -> None:
    self.window_size = w
    pass

  def read_data(self):
    DATADIR="dataset/"
    samples = []
    annotations = []
    for i in range(0, 73):
      if i == 14:
        continue
      sample = DATADIR + "samples_" + str(i+1) + ".csv"
      annotation = DATADIR + "annotation_" + str(i+1) + ".csv"
    
      with open(sample) as f:
        content = [line.strip().split(',') for line in f] 
      content = np.array(content)
      content = content[2:,1]
      samples.append(content)
      
      with open(annotation) as f:
        content2 = [line.strip().split(',') for line in f] 
      content2 = np.array(content2)
      content2 = content2[1:,0]
      annotations.append(content2)
    self.allSamples = np.array(samples)
    self.allAnnotations = np.array(annotations)
    # print(self.allSamples.shape, self.allAnnotations.shape)

  def one_hot(self, value):
    value = float(value)
    if value == 0:
      x = [1,0,0,0]
    elif value == 1:
      x = [0,1,0,0]
    elif value == 2:
      x = [0,0,1,0]
    else:
      x = [0,0,0,1]
    return x

  def get_window_seq(self, sample, annot):
    w = self.window_size
    sampleSeqs = []
    sampleLabels = []
    for j in range(len(sample) - self.window_size + 1):
      window_seq = []
      for i in range(w):
        window_seq.append(float(sample[j + i]))
        if i != w-1:
          window_seq.append(float(annot[j + i]))
        else:
          y = self.one_hot(annot[j + i]) 
      sampleSeqs.append(window_seq)
      sampleLabels.append(y)
    return (sampleSeqs, sampleLabels)
      
  def get_all_patterns(self):
    yTrain = []
    xTrain = []
    xTest = []
    yTest = []
    for i in range(len(self.allSamples)):
      sample = self.allSamples[i]
      annotation = self.allAnnotations[i]
      sampleXs, sampleYs = self.get_window_seq(sample, annotation)
      numTrainSample = round(len(sampleXs) * .8)
      xTrain.extend(sampleXs[:numTrainSample])
      yTrain.extend(sampleYs[:numTrainSample])
      xTest.extend(sampleXs[numTrainSample:])
      yTest.extend(sampleYs[numTrainSample:])    
    xTrain.extend(xTest)
    yTrain.extend(yTest)
    self.X = xTrain
    self.Y = yTrain 
    # print(self.X.shape, self.Y.shape)

  def learn_elman(self, xTrain, yTrain, ep):
    model = NeuralNet ()
    input_nodes = len(xTrain[0])
    hidden_nodes = self.window_size
    output_nodes = len(yTrain[0])
    model.init_layers ( input_nodes, [ hidden_nodes ], output_nodes, ElmanSimpleRecurrent () )
    model.randomize_network ()
    model.layers[1].set_activation_type ('sigmoid')
    model.set_learnrate (0.05)
    model.set_all_inputs (xTrain)
    model.set_all_targets (yTrain)
    length = len(xTrain)
    learn_end_point = int( length * 0.8)
    model.set_learn_range (0, learn_end_point )
    model.set_test_range ( learn_end_point + 1, length -1)
    model.learn( epochs = ep , show_epoch_results=True , random_testing=False)
    mse = model.test()
    y_real = model.test_targets_activations
    y_real = np.array(y_real)
    print("accuracy test of elman rnn is:" ,cal_accuracy(y_real), "and mse is:", np.round(mse ,6), "with window size:", self.window_size, "and epoc number:", ep )

  def learn_narx(self, X, Y, ep):
    input_nodes = len(X[0])
    hidden_nodes = self.window_size
    output_nodes = len(Y[0])
    output_order = 3
    incoming_weight_from_output = .5
    input_order = 2
    incoming_weight_from_input = .5

    net = NeuralNet()
    net.init_layers(input_nodes, [hidden_nodes], output_nodes,
            NARXRecurrent(
                output_order,
                incoming_weight_from_output,
                input_order,
                incoming_weight_from_input))

    net.randomize_network()
    net.layers[1].set_activation_type('sigmoid')
    net.set_learnrate (0.05)
    net.set_all_inputs(X)
    net.set_all_targets(Y)
    length = len(X)
    learn_end_point = int(length * 0.8)
    net.set_learn_range (0, learn_end_point )
    net.set_test_range (learn_end_point + 1,length -1)
    net.learn(epochs=ep ,show_epoch_results =True ,random_testing = False )
    mse = net.test()
    y_real = net.test_targets_activations
    y_real = np.array(y_real)
    print("accuracy test of narx rnn is:" ,cal_accuracy(y_real),"percent and mse is:", np.round(mse ,6), "with window size:", self.window_size, "and epoc number:", ep )

  def main(self):
    self.read_data()
    self.get_all_patterns()
    # self.learn_elman(self.X[50:], self.Y[50:], ep=1)
    self.learn_narx(self.X[100:], self.Y[100:], ep=1)

def cal_accuracy(y):
  acc = 0
  for tup in y:
    diff = abs(tup[1] - np.array([1,1,1,1]))
    ind1 = np.argmin(diff)
    ind2 = np.where(tup[0] == 1.0)[0]
    if ind1==ind2:
      acc += 1
  return(round(acc/len(y)*100, 2))

start = time.time()
r = Rnn(w=5)
r.main()
end = time.time()
s = (end - start)
conversion = datetime.timedelta(seconds=s)
print("Runtime of the program is", conversion)

# start = time.time()
# r = Rnn(w=11)
# r.main()
# end = time.time()
# s = (end - start)
# conversion = datetime.timedelta(seconds=s)
# print("Runtime of the program is", conversion)

# start = time.time()
# r = Rnn(w=21)
# r.main()
# end = time.time()
# s = (end - start)
# conversion = datetime.timedelta(seconds=s)
# print("Runtime of the program is", conversion)




