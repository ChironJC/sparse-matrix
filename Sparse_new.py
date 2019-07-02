#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:40:42 2019

@author: jiachenlu
"""
import numpy as np
import math, random
import tensorflow as tf
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



def genSpar(n,sparsity):
    # generate a random sparse matrix with nonzeros labled 1
    A = np.identity(n)
    for i in range(0,math.floor(n**2*sparsity)):
        [p,q] = [random.randint(0,n-1),random.randint(0,n-1)]
        A[p][q] = 1
        A[q][p] = 1
    for i in range(n):
        A[i][i] = i
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
    order = list(range(n))
    return adj, xadj, order

def getElimTree(adj, xadj):
    # get elimination tree for A
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

def getConElimTree(adj, xadj):
    # connect seperate elimination tree
    C = []
    v = getElimTree(adj, xadj)
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

def genConSpar(n, sparsity):
    #A connected graph might be easy for coding, so I connect the graoh here
    A = genSpar(n, sparsity)
    adj, xadj = upperSpar(A)
    C = getConElimTree(adj, xadj)
    for i in range(len(C)-1):
        m = list(C[i])[0]
        n = list(C[i+1])[0]
        A[m][n] = 1
        A[n][m] = 1
    return upperSpar(A)

def getFill(adj, xadj):
    # get the positions where fills happen where the entries are labeled 2
    v = getElimTree(adj, xadj)
    n = len(xadj)
    nadj = adj.copy()
    nxadj = xadj.copy()
    for i in range(len(xadj)-1):
        if xadj[i] == xadj[i+1]: pass
        for j in range(xadj[i],xadj[i+1]):
            r = adj[j]
            while v[r].parent != i:
                r = v[r].parent
                if r not in nadj[nxadj[i]:nxadj[i+1]]:
                    if nadj[nxadj[i+1]-1] < r:
                        nadj.insert(nxadj[i+1], r)
                    else:
                        for k in range(nxadj[i],nxadj[i+1]-1):
                            if (nadj[k] < r and r < nadj[k+1]):
                                nadj.insert(k+1, r)
                    for k in range(i+1,n):
                        nxadj[k] = nxadj[k]+1
    return nadj, nxadj

def genRanPerm(n):
    #I don't have any strategy on reordering for now, so I just randomly
    # generate one permutation
    L = [i for i in range(n)]
    L0 = []
    for i in range(n):
        k = random.randint(0,len(L)-1)
        j = L.pop(k)
        L0.append(j)
    return L0

def reorder(adj, xadj, order, current_order):
    #To get the matrix after reordering
    count = 0
    nadj = []
    nxadj = []
    for i in range(len(order)):
        temp = []
        nxadj.append(count)
        i_index = order[i]
        for j in range(i):
            j_index = order[j]
            if i_index < j_index:
                if (i_index in adj[xadj[j_index]:xadj[j_index+1]]):
                    temp.append(j)
                    count += 1
            if i_index > j_index:
                if j_index in adj[xadj[i_index]:xadj[i_index+1]]:
                    temp.append(j)
                    count += 1
        nadj.extend(sorted(temp))
    nxadj.append(count)
    temp = [current_order[i] for i in order]
    return nadj, nxadj, temp

def reorderBySeparator(adj, xadj,separator, current_order=None):
    #put seperators at the end
    if current_order == None:
        current_order = list(range(len(xadj)-1))
    L = [i for i in current_order if i not in separator]
    L.extend(list(separator))
    adj, xadj, new_order = reorder(adj, xadj, L, current_order)
    rxadj = xadj[:-len(separator)]
    radj = adj[:rxadj[-1]]
    C = getConElimTree(radj, rxadj)
    L = []
    for c in C:
        L.extend(list(c))
    L.extend(list(separator))
    return adj, xadj, new_order
'''
def RCM(A, root):
    #reversed Cuthill McKee method return permutation and a pseudo-pripheral vertice
    [adj, xadj] = graRepSpar(A)
    new_root = {root}
    perm = [root]
    root = set()
    while new_root - root != set() or new_root == set():
        adding = set()
        sorted_adding = []
        root = new_root.copy()
        for i in root:
            for j in range(xadj[i], xadj[i+1]):
                v = adj[j]
                if v not in root:
                    adding.add(v)
        if adding == set():
            break
        dict_adding = {i: xadj[i+1]-xadj[i] for i in adding}
        sorted_adding = sorted(dict_adding)
        perm.extend(sorted_adding)
        v = sorted_adding[0]
        new_root = new_root|adding
    return list(reversed(perm)),v

def edgeSparator(A, V):
    #from separation generate edge separator
    edge = []
    [adj, xadj] = upperSpar(A)
    for v in V:
        for i in v:
            for j in range(xadj[i], xadj[i+1]):
                m = adj[j]
                if m not in v:
                    edge.append((i, m))
    return edge

def verticesSeparator(V):
    #from edge aeparator generate separator
    deg = {}
    separator = set()
    for edge in V:
        (i, j) = edge
        for m in [i,j]:
            try:
                deg[m] += 1
            except KeyError:
                deg[m] = 1
    for edge in V:
        (i, j) = edge
        if i in separator or j in separator:
            pass
        if deg[i] > deg[j]:
            separator.add(i)
        else:
            separator.add(j)
#        print(separator)
    return separator

def printMatrix(A):
    #print the matrix and return the number of fills
    count = 0
    n = len(A)
    for i in range(n):
        for j in range(n):
            if i == j:
                print('%2s' % int(A[i][j]), end="")
            elif A[i][j] == 1: print('%2s' % "*", end="")
            elif A[i][j] == 2: 
                print('%2s' % "+", end="")
                count = count+1
            else: print('%2s' % " ", end="")
        print()
    return count

def minDegree(A):
    #Minimun degree method
    [adj, xadj] = graRepSpar(A)
    degree = {}
    for i in range(len(A)):
        degree[i] = xadj[i+1]-xadj[i]
    L = sorted(degree.items(), key=lambda kv: kv[1])
    L = [i for (i, j) in L]
#    print(L)
    return L
'''
def countFill(adj, xadj):
    #number of fill
    nadj, nxadj = getFill(adj, xadj)
    fill = nxadj[-1] - xadj[-1]
    return fill

def findNeighbour(adj, xadj, V):
    neighbour=adj[xadj[V]:xadj[V+1]]
    for i in range(V+1,len(xadj)-1):
        if V in adj[xadj[i],xadj[i+1]]:
            neighbour.apend(i)
    return set(neighbour)

def checkSeparator(adj, xadj, separator):
    if len(separator) == 0:
        return False
    check = True
    adj, xadj, new_order = reorderBySeparator(adj, xadj, separator)
    rxadj = xadj[:-len(separator)]
    radj = adj[:rxadj[-1]]
    C = getConElimTree(radj, rxadj)
    if len(C) == 1:
        check = False
    return check

def reward(adj, xadj, V):
    #return reward = Π(len(separated parts)) / len(separator)
    reward = 1 / len(V)
    adj, xadj, n = reorderBySeparator(adj, xadj, V)
    sadj = adj[0:xadj[-len(V)]]
    sxadj = xadj[0:-len(V)]
    C = getConElimTree(sadj, sxadj)
    for s in C:
        reward = reward * len(s)
    return reward

def step(action, adj, xadj, separator, current_order):
    if action in separator:
        separator.pop(action)
        if not checkSeparator(adj, xadj, separator):
            neighbour = findNeighbour(adj, xadj, action)
            separator += neighbour           
    else:
        separator += {action}

    reward0 = reward(adj, xadj, separator)
    adj, xadj, current_order = reorderBySeparator(
                    adj, xadj, separator, current_order)
    return adj, xadj, separator, current_order, reward0

#here I generate random connected sparse matrix and compute the fills
'''
n = 10
sparsity = 0.05
separator = set{}

adj, xadj = genConSpar(n, sparsity)

print(adj, xadj)
#print(countFill(A))
print(getFill(adj,xadj))
L = genRanPerm(n)

'''
'''
print(printMatrix(B))
root = 0
L, root = RCM(A, root)
L, root = RCM(A, root)
print(L)
C = reorderByPerm(A, L)
D = getFill(C)
print(printMatrix(D))
V=[set(range(0,int(n/2))), set(range(int(n/2),n))]
E=edgeSparator(D, V)
#print(E)
S = verticesSeparator(E)
#print(S)
F = reorderBySeparator(C,S)
F = getFill(F)
print(printMatrix(F))
reward = observe(C,S)
print(reward)
L = minDegree(A)
G = reorderByPerm(A, L)
G = getFill(G)
print(printMatrix(G))

#visualize graph
rows, cols = np.where(A == 1)
edges = zip(list(rows),list(cols))
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr, node_size=n)
plt.show()
'''