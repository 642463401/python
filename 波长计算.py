import math
while True:
    h=eval(input("请输入水深h:"))
    T=eval(input("请输入周期:"))
    L2=9.81*T*T/2/3.14
    L1=int(L2*10000)
    for i in range(1,L1):
        l = 9.81*T*T/2/3.14*math.tanh(2*3.14*h/(i/100))
        l1=int(l*10000)
        #print(l1)
        if i==l1:
            print(i/10000)
            break
