#coding=utf-8

from numpy import *
import os

# X中的每一个特征应该是连续的,否则你应该使用其他方法进行缩放
def meanNorm(X):
    X_norm=X

    #Gaussian分布归一化
    mu=mean(X,axis=0)
    sigma=std(X,axis=0)
    fileOut=open(os.path.join(os.path.dirname(__file__),'Gaussian.txt'),'w')
    # 将每一列数据的期望和方差都写入到文件中
    row,column=mu.shape
    for i in xrange(0,column):
        write_data=str(mu[0,i])
        if i!=column-1:
            write_data+='\t'
        else:
            write_data+='\n'
        fileOut.write(write_data)

    row,column=sigma.shape
    for j in xrange(0,column):
        write_data=str(sigma[0,j])
        if j!=column-1:
            write_data+='\t'
        else:
            write_data+='\n'
        fileOut.write(write_data)

    fileOut.close()

    m=size(X,0)
    mu_matrix=dot(ones([m,1]),mu)
    sigma_matrix=dot(ones([m,1]),sigma)

    #think divide according to multiply
    X_norm=divide(X-mu_matrix,sigma_matrix)
    return  X_norm