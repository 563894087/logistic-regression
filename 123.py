import numpy as np
import matplotlib.pyplot as plt
import sympy

class Dot():
    def __init__(self,x1,x2,value):
        self.x1 = x1
        self.x2 = x2
        self.value = value

dot_set = []
for i in range(200):
    if i <100:
        dot_x1 = np.random.uniform(-3,3)
        dot_x2 = -dot_x1**2 + 1 + np.random.uniform(-3,0)
        dot_example = Dot(dot_x1,dot_x2,0)
        dot_set.append(dot_example)
    else:
        dot_x1 = np.random.uniform(-3, 3)
        dot_x2 = -dot_x1 ** 2 + 1 + np.random.uniform(1,3)
        dot_example = Dot(dot_x1,dot_x2,1)
        dot_set.append(dot_example)

x1_negative = np.array([])
x2_negative = np.array([])
x1_positive = np.array([])
x2_positive = np.array([])
for i in range(100):
    x1_negative = np.append(x1_negative,dot_set[i].x1)
    x2_negative = np.append(x2_negative,dot_set[i].x2)
for i in range(101,200):
    x1_positive = np.append(x1_positive,dot_set[i].x1)
    x2_positive = np.append(x2_positive,dot_set[i].x2)
"""------------------------上为生成数据集--------------------------------"""

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def cost_func(z,dot_set):
    result = 0
    for i in range(200):
        result += (dot_set[i].value * np.log(sigmoid_func(z[:,i])) +
                (1-dot_set[i].value)*np.log(1-sigmoid_func(z[:,i])))
    return (-1/200)*result

def deritative(theta_j,x_j,z,value,alpha): # 除了theta_j和alpha之外都是矩阵的形式
    result = 0
    #a = x_j.shape
    #b = value.shape
    for i in range(200):
        result += ((sigmoid_func(z[:,i]) - value[i]) * x_j[i])
    theta_j -= alpha * (result * 1/200)
    return theta_j

def make_z_matrix(theta,x_matrix):# 1*n的矩阵关于h(g(z))中的z
    z_matrix = np.array([])
    temp = theta.reshape(1,4)
    z_matrix = np.dot(temp,x_matrix)
    return z_matrix

def make_x_matrix(dot_set): # j*n的矩阵关于x上标i
    x_matrix = np.array([1,dot_set[0].x1**2,dot_set[0].x1,dot_set[0].x2])
    x_matrix.shape = 4,1
    for i in range(1,200):
        temp = np.array([1,dot_set[i].x1**2,dot_set[i].x1,dot_set[i].x2])
        temp.shape = 4,1
        x_matrix = np.hstack((temp,x_matrix))
    return x_matrix

#开始进行一些关键字矩阵的初始化以及梯度下降
x_matrix = make_x_matrix(dot_set)

theta0 = np.random.uniform(-2,2)
theta1 = np.random.uniform(-2,2)
theta2 = np.random.uniform(-2,2)
theta3 = np.random.uniform(-2,2)
theta = np.array([theta0,theta1,theta2,theta3])


z_matrix = make_z_matrix(theta,x_matrix)

value_matrix = np.array([])
for i in range(200):
    value_matrix = np.append(value_matrix,dot_set[i].value)

while(cost_func(z_matrix,dot_set) > 0.2):
    temp = np.array([])
    for i in range(4):
        result = deritative(theta[i],x_matrix[i,:],z_matrix,value_matrix,0.001)
        temp = np.append(temp,result)
    for i in range(4):
        theta[i] = temp[i]
    print("theta0=%f\ttheta1=%f\ttheta2=%f\ttheta3=%f\tcost=%.18f"
            %(theta[0],theta[1],theta[2],theta[3],cost_func(z_matrix,dot_set)))
    z_matrix = make_z_matrix(theta,x_matrix)

plot_x1 = np.linspace(-3,3,200)
plot_x2 = np.array([])
for i in range(200):
    tempx2 = sympy.Symbol("tempx2")
    a = sympy.solve([theta[0] + theta[1]*(plot_x1[i]**2)
                                          + theta[2]*plot_x1[i] + theta[3]*tempx2],[tempx2])
    plot_x2 = np.append(plot_x2,a[tempx2])
    print(i)

plt.figure(figsize=(10,6))
plt.scatter(x1_negative,x2_negative,color = "red")
plt.scatter(x1_positive,x2_positive,marker= ">")
plt.plot(plot_x1,plot_x2,label = "$logistic regression$",color = "yellow")
plt.xlabel("x1 axis")
plt.ylabel("x2 axis")
plt.legend()
plt.show()
