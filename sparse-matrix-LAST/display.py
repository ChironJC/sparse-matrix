#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:57:45 2019

@author: jiachenlu
"""
# this file is used to visualize th e separator in a 5*5 grid

import matplotlib.pyplot as plt

def findCo(i):
    x = i % 5
    y = 4 - (i-i%5) / 5
    return x,y

f = open('data5.txt')
s1 = 10
s2 = 5
c = 10
f.readline()
plt.show()
i=0
while True:
    fig, ax = plt.subplots()
    for i in range(25):
        x,y = findCo(i)
        plt.scatter(x, y, color='red')
        
    line = f.readline()
    i+=1
    
    line = str(line)
    print(line)
    if line.find('random') != -1:
        c = 'green'
    else:
        c = 'blue'
    start = line.find('{')
    end = line.find('}')
    separator_str = line[start+1:end]
    print(separator_str)
    temp = separator_str.split(',')
    separator = [int(i) for i in temp]
    for i in separator:
        x, y = findCo(i)
        ax.scatter(x, y, color=c)
    plt.show()
f.close()