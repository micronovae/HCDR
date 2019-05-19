import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq  

f1 = pd.read_excel('data1.xlsx')
f2 = pd.read_excel('data1.xlsx',
  sheetname = 1)
f3 = pd.read_excel('data1.xlsx',
  sheetname = 2)

makCap = f3.iloc[:,2]
makCap = makCap/sum(makCap)

f1 = f1.dropna()
f2 = f2.dropna()
f3 = f3.dropna()
f1 = f1.drop(['Month'],axis=1)

def retVol(f1):

  sta = []
  for i in range(100):
    data = np.array([0]*len(f1.iloc[:,0]))
    for j in range(i+1):
      data = data + f1.iloc[:,j]
    var = np.std(data/(i+1))
    sta.append(var)
    # print(data[0])

  return sta

def func(p, rm, rf = 0.015):
  alf,beta=p
  return alf+beta*(rm-rf)

def errorfunc(p, rm, ri):
  return func(p, rm)-ri

def sigIndModel(ri, rm):
  p0 = [0,0]
  Para=leastsq(errorfunc,p0,args=(rm,ri))
  return Para[0]

def rmfun(makCap, f2):

  rm = []
  for i in range(len(f2.iloc[:,0])):
    ri = f2.iloc[i,:]
    rm.append(np.dot(ri,makCap))
  return rm

def main():

  rm = np.array(rmfun(makCap, f2))
  alf = []
  beta = []
  pri = []

  for i in range(len(f2.iloc[0,:])):
    Para = sigIndModel(f2.iloc[:,i],rm)
    alf.append(Para[0])
    beta.append(Para[1])
    pri.append(np.mean(f2.iloc[:,i]))

  alf = pd.Series(alf)
  index = alf[(alf>np.mean(alf))].index.tolist()
  print(alf[alf<0])
 
  NewMakCapTot = 0
  NewMakCap = [0]*len(f3.iloc[:,0])

  for i in index:
    NewMakCapTot = NewMakCapTot + f3.iloc[i,2]
  for i in index:
    NewMakCap[i] = (f3.iloc[i,2]/NewMakCapTot)

  newRm = np.array(rmfun(NewMakCap, f2))

  # print(f1.iloc[:,0])
  # print(np.mean(f2.iloc[0,:]))
  # print(newRm[0])

  print(np.mean(rm))
  print(np.mean(newRm))

  # print(np.std(rm))
  # print(np.std(newRm))
  # print(beta)

  # plt.plot(rm,f2.iloc[:,0],'ro')
  # riP = (rm-0.03)*beta[0]+alf[0]
  # plt.plot(rm,riP,'b',lw=2)
  # plt.grid(True) 
  # plt.title('ADMIRAL GROUP RETURN GRAPH')
  # plt.xlabel('Market Return')
  # plt.ylabel('Stock Return')
  # plt.show()

  # plt.figure(2)
  # plt.hist(riP-f2.iloc[:,0],14,alpha=0.75)
  # plt.title('Residual distribution of ADMIRAL GROUP')
  # plt.xlabel('Residual')
  # plt.ylabel('Numbers')
  # plt.show()
  


  # data = {'Alpha': alf, 'Beta':beta, 'omega': makCap ,
  # 'newOmega':NewMakCap, 'meanRet':pri}
  # data = pd.DataFrame(data)  
  # data.to_excel("sample.xlsx")


if __name__ == '__main__':
  main()
  # sta1 = retVol(f1)
  # sta2 = retVol(f2)
  # plt.plot(sta1[:100],'r',lw=2,label='simulated data')
  # plt.plot(sta2[:100],lw=2,label='historic data')
  # plt.legend()
  # plt.grid(True) 
  # plt.xlabel('Number of stocks in the portfolio')
  # plt.ylabel('Volatility')
  # plt.title('Impact on Portfolio Volatility on number of stocks held')
  # plt.show()
  
