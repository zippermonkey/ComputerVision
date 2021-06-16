import numpy as np
import os

# 加载数据集
data_dir = "./lab/data/MNIST/raw/"
train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
test_labels_path = os.path.join(data_dir,'t10k-labels-idx1-ubyte')
test_images_path = os.path.join(data_dir,'t10k-images-idx3-ubyte')

print(os.getcwd())

with open(train_images_path) as fd:
    loaded = np.fromfile(file=fd , dtype=np.uint8)

trX = loaded[16:].reshape((60000, 784)).astype(np.float)

with open(train_labels_path) as fd:
    loaded = np.fromfile(file=fd , dtype=np.uint8)

trY = loaded[8:].reshape((60000,  1)).astype(np.float)

with open(test_images_path) as fd:
    loaded = np.fromfile(file=fd , dtype=np.uint8)

teX = loaded[16:].reshape((10000, 784)).astype(np.float)

with open(test_labels_path) as fd:
    loaded = np.fromfile(file=fd , dtype=np.uint8)

teY = loaded[8:].reshape((10000,  1)).astype(np.float)

# 定义onehot 
def one_hot(x, num_class=None):
    """
    返回onehot向量

    参数:
        x -- 标签
        num_class -- 类别数目

    返回:
        onehot编码
    """
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((len(x), num_class))
    for i,v in enumerate(x):
      ohx[i][int(v)] = 1
    return ohx

# 对数据集进行normalize
def norm(x):
    mean = x.mean()         #计算平均数
    deviation = x.std()     #计算标准差
    return (x - mean) / deviation  

# 添加bias列
def add_bias(x):
    m = len(x)
    return np.concatenate((np.ones((m,1)),x),1)

# sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# sigmoid 的导数
def derivative_sigmoid(z):
    return z*(1-z)

# relu
def relu(x):
    return (abs(x) + x) / 2

def derivative_relu(z):
    z[z>0]=1
    z[z<=0]=0
    return z

# softmax

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

# LOSS
def cross_entropy_error(y,t):
    delta=1e-7  #添加一个微小值可以防止负无限大(np.log(0))的发生。
    return -np.sum(t*np.log(y+delta))

# 计算准确率
def accuracy(y,t):
    # argmax 
    sum = 0
    for i,v in enumerate(y):
        if (v==t[i]).all():
            sum+=1
    return sum

# 打乱 
def shuffle(X, Y):
    """
    打乱数据集(X，Y)

    参数:
        X -- 图像数据，float32类型的矩阵
        Y -- 独热（one-hot）标签，uint8类型的矩阵

    返回:
        shuffles -- 字典，{"X_shuffle": X_shuffle, "Y_shuffle": Y_shuffle}
    """

    #取数据集大小
    m =len(X)

    #随机生成一个索引顺序
    permutation = list(np.random.permutation(m))

    #把X,Y打乱成相同顺序
    X_shuffle = X[permutation,:]
    Y_shuffle = Y[permutation,:]

    #打乱的数据集存在字典里
    shuffles = {"X_shuffle": X_shuffle, "Y_shuffle": Y_shuffle}
    return shuffles

# 获取minibatch
def get_mini_batches(X, Y, mini_batch_size = 60):
    """
    把数据集按照迷你批大小进行分割

    参数:
        X -- 图像数据，float32类型的矩阵
        Y -- 独热（one-hot）标签，uint8类型的矩阵
        mini_batch_size -- 迷你批大小

    返回:
        mini_batches -- 元素为（X,Y）元组的列表
    """
    #调用刚才的函数
    shuffles = shuffle(X, Y)

    #取数据集大小
    num_examples = shuffles["X_shuffle"].shape[0]

    #计算完整迷你批的个数
    num_complete =  num_examples // mini_batch_size

    #建立一个空列表，存储迷你批
    mini_batches = []

    #分配完整的迷你批
    for i in range(num_complete):
        mini_batches.append([shuffles["X_shuffle"]\
                            [i*mini_batch_size:(i+1)*mini_batch_size,:], \
                            shuffles["Y_shuffle"]\
                            [i*mini_batch_size:(i+1)*mini_batch_size,:]])

    #如果需要的话，分配不完整的迷你批
    if 0 == num_examples % mini_batch_size:
        pass
    else:
        mini_batches.append([shuffles["X_shuffle"]\
                            [num_complete*mini_batch_size:,:], \
                            shuffles["Y_shuffle"]\
                            [num_complete*mini_batch_size:,:]])

    return mini_batches


# train

# theta
theta1 = np.random.randn(785,100)
theta2 = np.random.randn(101,50)
theta3 = np.random.randn(51,10)


# data
trX = norm(trX)
trX = add_bias(trX)

teX = norm(teX)
teX = add_bias(teX)

trY = one_hot(trY,10)
teY = one_hot(teY,10)


mini = get_mini_batches(trX, trY, 60)


def train(mini_batches,theta1,theta2,theta3,itrs,lr = 0.01):
    for i in range(itrs):
        data_size = 0
        correct_num = 0
        all_loss = 0
        r1,r2,r3=0,0,0
        eps = 1e-7
        for j, data in enumerate(mini_batches):
            X = data[0]
            Y = data[1]
            m = len(X)
            data_size+=m
            # print(m)
            z1 = X.dot(theta1)  # m * 1000
            # a1 = sigmoid(z1)
            a1 = relu(z1)
            # add bias
            a1 = np.concatenate((np.ones((m,1)),a1),1)
            z2 = a1.dot(theta2)  # m * 1000
            # a2 = sigmoid(z2)  
            a2 = relu(z2)  

            a2 = np.concatenate((np.ones((m,1)),a2),1)
            z3 = a2.dot(theta3)  # m * 200
            a3 = softmax(z3) 
            
            # loss
            loss = cross_entropy_error(a3, Y)
            all_loss += loss
            # print(loss)
            
            # pred_y
            pred_y = np.argmax(a3,1).reshape((m,1))
            pred_y = one_hot(pred_y,10)

            correct_num += accuracy(pred_y,Y)

            # back
            
            dz3 = a3-Y
            dw3 = a2.T.dot(dz3)/m
            

            # dz2 = dz3.dot(theta3.T) * derivative_sigmoid(a2)
            dz2 = dz3.dot(theta3.T) * derivative_relu(a2)
            dw2 = a1.T.dot(dz2)/m
            dw2 = dw2[:,1:]

            dz2 = dz2[:,1:]

            # dz1 = dz2.dot(theta2.T) * derivative_sigmoid(a1)
            dz1 = dz2.dot(theta2.T) * derivative_relu(a1)
            dw1 = X.T.dot(dz1)/m
            dw1 = dw1[:,1:]

            r1 = r1+ dw1*dw1
            r2 = r2+ dw2*dw2
            r3 = r3+ dw3*dw3
 
            theta3= theta3 - lr*dw3/(np.sqrt(r3)+eps)
            theta2= theta2 - lr*dw2/(np.sqrt(r2)+eps)
            theta1= theta1 - lr*dw1/(np.sqrt(r1)+eps)
        acc = correct_num/data_size
        print(str(i)+ ":   "+str(all_loss)  + "   acc: " +str(acc))
    return theta1,theta2,theta3, all_loss,acc  


theta1,theta2,theta3, train_loss,train_acc  =  train(mini,theta1,theta2,theta3,itrs = 15,lr = 0.01)
print("train_loss: " + str(train_loss))
print("train_accuracy: " + str(train_acc))

# test

def test(theta1,theta2,theta3,X,Y):
    m = len(X)
    z1 = X.dot(theta1)  # m * 1000
    # a1 = sigmoid(z1)
    a1 = relu(z1)
    # add bias
    a1 = np.concatenate((np.ones((m,1)),a1),1)
    z2 = a1.dot(theta2)  # m * 1000
    # a2 = sigmoid(z2)  
    a2 = relu(z2)  

    a2 = np.concatenate((np.ones((m,1)),a2),1)
    z3 = a2.dot(theta3)  # m * 200
    a3 = softmax(z3) 


    # loss
    loss = cross_entropy_error(a3, Y)
    # print(loss)

    # pred_y
    pred_y = np.argmax(a3,1).reshape((m,1))
    pred_y = one_hot(pred_y,10)
    acc = accuracy(pred_y,Y)/m
    return loss, acc

test_loss,test_acc = test(theta1,theta2,theta3,teX,teY)
print("test_loss: " + str(test_loss))
print("test_accuracy: " + str(test_acc))

