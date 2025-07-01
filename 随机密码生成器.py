import random
x=int(input())
random.seed(x) 
s = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
mmgs=int(input())
mmcd=int(input())
for z in range(mmgs):
    b=''
    for i in range(mmcd):
        a=random.choice(s)
        b=b+a
    print(b)