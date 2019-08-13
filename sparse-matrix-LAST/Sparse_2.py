#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:17:35 2019

@author: jiachenlu
"""


import numpy as np
import math, random

class node:
    # for elimination tree
    def __init__(self,index):
        self.parent = index    #parent
        self.ancester = None   #root
        self.index = index     #index
        self.children = set() #set of all the children
        
    def setParent(self,n): #set a parent
        self.parent = n
        self.ancester = n.ancester
        n.children.add(self)
    
    def printTree(self):  #elimination tree display
        print(self.index)
        for i in self.children:
            i.printTree()

def genSpar(n,sparsity):
    # generate a random n*n sparse matrix with sparsity=sparsity 
    A = np.identity(n)
    for i in range(0,math.floor(n**2*sparsity)):
        [p,q] = [random.randint(0,n-1),random.randint(0,n-1)]
        A[p][q] = 1
        A[q][p] = 1
    for i in range(n):
        A[i][i] = i
    m = random.randint(0,n-2)
    A[n-1][m] = 1
    A[m][n-1] = 1
    return A

def upperSpar(A):
    # Store the upper part of a sparse matrix into CVS vectors adj and xadj
    n = len(A)
    adj = []
    xadj = []
    count = 0
    for i in range(n):
        xadj.append(count)
        for j in range(i):
            if A[i][j] == 1:
                adj.append(j)
                count += 1
    xadj.append(count)
    return adj, xadj

def getElimTree(adj, xadj):
    # get elimination tree from CVS vectors
    # go through all the vertices, if one is adjacent to another, set the paraent
    # of the later one as the earlier one
    v = []
    for i in range(len(xadj)-1):
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

def getConSpar(adj, xadj):
    # since the sparse matirx is randomly generated, it might be disconnected.
    # the graph is connected is this function.
    # find the elimination tree, if the graph is disconnected, the elimination tree
    # will bw disconnected.connect each root of the tree.
    v = getElimTree(adj, xadj)
    temp = []
    for i in v:
        if i.parent == i.index :
            temp.append(i.index)
    for i in temp:
        if i == len(xadj)-2:
            continue
        adj.insert(xadj[i+2], i)
        for j in range(i+2,len(xadj)):
            xadj[j] += 1
    return adj, xadj

def sparseToDense(adj, xadj):
    # generate dense matrix from CVS vector
    n = len(xadj)-1
    A = np.identity(n)
    for i in range(n):
        for j in range(xadj[i], xadj[i+1]):
            r = adj[j]
            A[i][r] = 1
            A[r][i] = 1
        A[i][i] = i
    return A

def printMatrix(A):
    #print the dense matrix
    n = len(A)
    count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                print('%2s' % i, end="")
            elif A[i][j] == 1: print('%2s' % "*", end="")
            elif A[i][j] == 2: 
                print('%2s' % "+", end="")
                count = count+1
            else: print('%2s' % " ", end="")
        print()
    return count

def printSparse(adj, xadj):
    # print the sparse matrix from CVS vectors
    printMatrix(sparseToDense(adj,xadj))
    return None

def printFill(adj, xadj, nadj, nxadj):
    # print the spase matrix and its fill
    A = sparseToDense(adj, xadj)
    B = sparseToDense(nadj, nxadj)
    n = len(A)
    count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                print('%2s' % i, end="")
            elif A[i][j] == 1: print('%2s' % "*", end="")
            elif A[i][j] != B[i][j]:
                print('%2s' % "+", end="")
                count = count+1
            else: print('%2s' % " ", end="")
        print()
    return count
        

def getFill(adj, xadj):
    # get the positions where fills happen. the positions are labeled 2.
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

def countFill(adj, xadj):
    # count the number of fill
    nadj, nxadj = getFill(adj, xadj)
    count = nxadj[-1]-xadj[-1]
    return count

def reorder(adj, xadj, permutation):
    # get the new CVS vectors after a reordering
    count = 0
    nadj = []
    nxadj = []
    for i in range(len(permutation)):
        temp = []
        nxadj.append(count)
        i_index = permutation[i]
        for j in range(i):
            j_index = permutation[j]
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
    return nadj, nxadj

def reorderBySeparator(adj, xadj, separator):
    # get the CVS vecters after one step of separating.
    
    # 1. find the separated groups.
    # set the neightbour of vertices in separator to empty, so that the elimination
    # tree will be disconnected.
    save_adj = adj.copy()
    save_xadj = xadj.copy()
    n = len(xadj)-1
    for i in separator: # delete the colomns
        k = xadj[i+1] - xadj[i]
        for m in range(k):
            adj.pop(xadj[i])
        for j in range(i+1,n+1):
            xadj[j] = xadj[j] - k
    for i in range(n): # delete the rows
        remove = []
        for j in range(xadj[i],xadj[i+1]):
            if adj[j] in separator:
                remove.append(j)
        count = 0
        for j in remove:
            adj.pop(j-count)
            count += 1
            for k in range(i+1, n+1):
                xadj[k] -= 1
    v = getElimTree(adj, xadj)
    
    # 2. find the new permutation.
    # group the separated parts and add the separator at the end to get the permutation
    temp = []
    permutation = []
    for i in range(len(v)): # find the root of each group
        if v[i].index == v[i].parent and v[i].index not in separator:
            temp.append(i)
    for i in temp: # add the subtrees to permutation
        w = {i}
        new_w = set()
        while len(w-new_w) != 0:
            new_w = w.copy()
            for i in new_w:
                for j in v[i].children:
                    w.add(j.index)
        permutation.extend(list(w))
    permutation.extend(list(separator)) # add separator
    
    # 3. reorder with permutation.
    adj, xadj = reorder(save_adj, save_xadj, permutation)
    return adj, xadj

def checkSeparator(adj, xadj, separator):
    # check if a separator is valid or not.
    # 1. find the separated groups.
    # set the neightbour of vertices in separator to empty, so that the elimination
    # tree will be disconnected.
    temp_adj = adj.copy()
    temp_xadj = xadj.copy()
    n = len(xadj)-1
    for i in separator:
        k = temp_xadj[i+1] - temp_xadj[i]
        for m in range(k):
            temp_adj.pop(temp_xadj[i])
        for j in range(i+1,n+1):
            temp_xadj[j] = temp_xadj[j] - k
    for i in range(n):
        remove = []
        for j in range(temp_xadj[i],temp_xadj[i+1]):
            if temp_adj[j] in separator:
                remove.append(j)
        count = 0
        for j in remove:
            temp_adj.pop(j-count)
            count += 1
            for k in range(i+1, n+1):
                temp_xadj[k] -= 1
    
    # 2. check the separator.            
    # if the separator is valid, there will be more than one root in the elimination tree
    v = getElimTree(temp_adj, temp_xadj)
    temp = []
    for i in range(len(v)):
        if v[i].index == v[i].parent and v[i].index not in separator:
            temp.append(i)
    del temp_adj, temp_xadj
    return len(temp) != 1

def findNeighbour(adj, xadj, V):
    # find the neighbour of each vertice in the sparse matrix
    neighbour = adj[xadj[V]:xadj[V+1]] # neighbours with index smaller than V
    for i in range(V+1,len(xadj)-1):   # neighbours with index bigger than V
        if V in adj[xadj[i]: xadj[i+1]]: neighbour.append(i)
    return set(neighbour)

def reward(adj, xadj, separator):
    # the reward function is given by -1/(product of sizes of the separated groups)
    temp_adj = adj.copy()
    temp_xadj = xadj.copy()
    n = len(xadj)-1
    
    # 1. find the separated groups.
    for i in separator:
        k = temp_xadj[i+1] - temp_xadj[i]
        for m in range(k):
            temp_adj.pop(temp_xadj[i])
        for j in range(i+1,n+1):
            temp_xadj[j] = temp_xadj[j] - k
    for i in range(n):
        remove = []
        for j in range(temp_xadj[i],temp_xadj[i+1]):
            if temp_adj[j] in separator:
                remove.append(j)
        count = 0
        for j in remove:
            temp_adj.pop(j-count)
            count += 1
            for k in range(i+1, n+1):
                temp_xadj[k] -= 1
    v = getElimTree(temp_adj, temp_xadj)
    del temp_adj, temp_xadj
    
    #2. find the roots
    temp = []
    for i in range(len(v)):
        if v[i].index == v[i].parent and v[i].index not in separator:
            temp.append(i)
            
    #calculate the reward
    reward = 1
    for i in temp:
        w = {i}
        new_w = set()
        while len(w-new_w) != 0:
            new_w = w.copy()
            for i in new_w:
                for j in v[i].children:
                    w.add(j.index)
        reward = reward*len(w)
    print(len(temp))
    reward = -1/reward
    return reward
    
def step(action, adj, xadj, separator):
    # do one step with a given action.
    ini_reward = reward(adj, xadj, separator)
    if action in separator:
        m = separator.remove(action)
        reward_bon = 0.05
        if checkSeparator(adj, xadj, separator) == False:
            neighbour = findNeighbour(adj, xadj, action)
            separator = separator.union(neighbour)
            separator.add(action)
            reward_bon = 0
    else:
        reward_bon = 0
    '''
    else:
        separator.add(action)
        reward_cons = 0.8
        if checkSeparator(adj, xadj, separator) == False:
            m = separator.remove(action)
            reward_cons = 0.5
    '''
    reward0 = (-ini_reward + reward(adj, xadj, separator)) + reward_bon
    return separator, reward0

def turn(ver):
    # turn the 5*5 grid 90 degree counter-clockwisely
    a1 = ver % 5
    a2 = (ver -a1)/5
    new_ver = 5*(4-a1) + a2
    return new_ver

def reflect(ver):
    # reflect the 5*5 grid
    a1 = ver % 5
    a2 = (ver -a1)/5
    new_ver = 5*a2 +(4-a1)
    return new_ver

def add(n_turn, n_reflect, ver):
    # do n_turn times of turn and n_reflect times of reflect
    for i in range(n_turn):
        ver = turn(ver)
    if n_reflect == 1:
        ver = reflect(ver)
    return ver

'''
A = [[1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1]]

print(A[2][1])
adj,xadj = upperSpar(A)
print(adj, xadj)
separator = {0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23}
#adj, xadj = reorderBySeparator(adj, xadj, separator)
action = 9
separator, reward = step(action, adj, xadj, separator)
print(separator)
print(findNeighbour(adj, xadj, 4))
'''