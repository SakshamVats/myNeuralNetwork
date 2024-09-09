from math import e
import numpy as np
import matplotlib.pyplot as plt

### config ###
SEED = 6
HIDDEN_NODES = 4
OUTPUT_NODES = 3
SAMPLES_EACH = 700
LEARNING_RATE = 10e-6
EPOCH = 50000
PRINT_INTERVAL = 200

#seed random generator
np.random.seed(SEED)

#create virtual images
cat_images = np.random.randn(SAMPLES_EACH,2) + np.array([0,-3])
mouse_images = np.random.randn(SAMPLES_EACH,2) + np.array([3,3])
dog_images = np.random.randn(SAMPLES_EACH,2) + np.array([-3,3])

#stack images to make input
feature_set = np.vstack([cat_images, mouse_images, dog_images])

#generate labels
labels = np.array([0]*SAMPLES_EACH + [1]*SAMPLES_EACH + [2]*SAMPLES_EACH)

#make them one-hot labels for output
one_hot_labels = np.zeros((SAMPLES_EACH*3, 3))
for i in range(SAMPLES_EACH*3):
    one_hot_labels[i, labels[i]] = 1

#plot data set
'''
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap='plasma', s=100, alpha=0.5)
plt.show()
'''

#activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

#neural net parameters
instances = SAMPLES_EACH * 3
attributes = 2
hidden_nodes = HIDDEN_NODES
output_labels = OUTPUT_NODES

wh = np.random.rand(attributes, hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes, output_labels)
bo = np.random.randn(output_labels)
lr = LEARNING_RATE

error_cost = []

for epoch in range(EPOCH):
    ### feed forward ###
    #phase 1
    zh = np.dot(feature_set, wh) + bh
    ah = sigmoid(zh)

    #phase 2
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)

    ### backpropagation ###
    #phase 1
    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_dwo = np.dot(dzo_dwo.T, dcost_dzo)
    dcost_dbo = dcost_dzo

    #phase 2
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set

    dcost_dwh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
    dcost_dbh = dah_dzh * dcost_dah

    #update weights and biases
    wh -= lr * dcost_dwh
    bh -= lr * dcost_dbh.sum(axis=0)

    wo -= lr * dcost_dwo
    bh -= lr * dcost_dbh.sum(axis=0)

    ### print error ###
    if epoch % PRINT_INTERVAL == 0:
        loss = np.sum(-1 * one_hot_labels * np.log(ao))
        print(f"Loss function value after iteration {epoch}: ", loss)
        error_cost.append(loss)


