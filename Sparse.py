#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:40:42 2019

@author: jiachenlu
"""
import numpy as np
import math, random
import matplotlib.pyplot as plt
import networkx as nx

class node:
    # for elimination tree
    def __init__(self,index):
        self.parent = index
        self.ancester = None
        self.index = index
        self.children = set()
        
    def setParent(self,n):
        self.parent = n
        self.ancester = n.ancester
        n.children.add(self)
    
    def printTree(self):
        print(self.index)
        for i in self.children:
            i.printTree()



def genSpar(n):
    # generate a random sparse matrix with nonzeros labled 1
    A = np.identity(n)
    for i in range(0,math.floor(n**2/10)):
        [p,q] = [random.randint(0,n-1),random.randint(0,n-1)]
        A[p][q] = 1
        A[q][p] = 1
    return A

def graRepSpar(A):
    # Store a sparse matrix
    n = len(A)
    adj = []
    xadj = []
    count = 0
    for i in range(0,n):
        xadj.append(count)
        for j in range(0,n):
            if A[i][j] == 1:
                adj.append(j)
                count += 1
    xadj.append(count)
    return [adj, xadj]

def upperSpar(A):
    # Store the upper part of a sparse matrix
    n = len(A)
    adj = []
    xadj = []
    count = 0
    for i in range(0,n):
        xadj.append(count)
        try:
            for j in range(0,i):
                if A[i][j] == 1:
                    adj.append(j)
                    count += 1
        except ValueError:
            pass
    xadj.append(count)
    return [adj, xadj]

def getElimTree(A):
    # get elimination tree for A
    [adj,xadj] = upperSpar(A)
    v = []
    for i in range(0,len(xadj)-1):
        n = node(i)
        v.append(n)
        for j in range(xadj[i],xadj[i+1]):
            m = adj[j]
            r = v[m]
            while r.parent != r.index:
                r = v[r.parent]
            r.parent = i
            v[i].children.add(r)
    return v

def getConElimTree(A):
    # connect seperate elimination tree
    C = []
    v = getElimTree(A)
    w = []
    for i in range(len(v)):
        if v[i].parent == i:
            w.append(i)
    for i in range(len(w)):
        new = {w[i]}
        Ctemp = set()
        while len(new-Ctemp) != 0:
            Ctemp = new.copy()
            for j in Ctemp:
                for r in v[j].children:
                    new.add(r.index)
        C.append(new)
    return C

def genConSpar(n):
    #A connected graph might be easy for coding, so I connect the graoh here
    A = genSpar(n)
    C = getConElimTree(A)
    for i in range(len(C)-1):
        m = list(C[i])[0]
        n = list(C[i+1])[0]
        A[m][n] = 1
        A[n][m] = 1
    return A

def getFill(A):
    # get the positions where fills happen where the entries are labeled 2
    v = getElimTree(A)
    B = A.copy()
    [adj, xadj] = upperSpar(A)
    for i in range(len(xadj)-1):
        if xadj[i] == xadj[i+1]: pass
        for j in range(xadj[i],xadj[i+1]):
            r = adj[j]
            while v[r].parent != i:
                r = v[r].parent
                if A[i][r] != 1:
                    B[i][r] = 2
                    B[r][i] = 2
    return B

def genRanPermu(n):
    #I don't have any strategy on reordering for now, so I just randomly
    # generate one permutation
    L = [i for i in range(n)]
    L0 = []
    for i in range(n):
        k = random.randint(0,len(L)-1)
        j = L.pop(k)
        L0.append(j)
    return L0

def reorderByPermu(A,L):
    #To get the matrix after reordering
    n = len(L)
    P = np.zeros((n,n))
    for i in range(len(L)):
        P[i][L[i]] = 1
    Pm = np.asmatrix(P)
    Am = np.asmatrix(A)
    Pt = np.transpose(Pm)
    A1 = Pm*Am*Pt
    A1 = np.asarray(A1)
    return A1

#here I generate random connected sparse matrix and compute the fills
n = 20
A = genConSpar(n)
for i in range(n):
    for j in range(n):
        if A[i][j] == 1 and i == j:
            print('%2s' % i, end="")
        elif A[i][j] == 1: print(" *", end="")
        else: print(" ", end=" ")
    print()
print(upperSpar(A))
B=getFill(A)
for i in range(n):
    for j in range(n):
        if B[i][j] != 0 and i == j:
            print('%2s' % i, end="")
        elif B[i][j] == 1: print('%2s' % "*", end="")
        elif B[i][j] == 2: print('%2s' % "+", end="")
        else: print('%2s' % " ", end="")
    print()

'''
#visualize graph
rows, cols = np.where(A == 1)
edges = zip(list(rows),list(cols))
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr, node_size=n)
plt.show()
'''