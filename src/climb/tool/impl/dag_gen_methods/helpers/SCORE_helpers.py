import os
import uuid
import numpy as np
import pandas as pd
import torch

def Stein_hess(X, eta_G, eta_H, s = None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    X, using first and second-order Stein identities
    """
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
    
    nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
    return -G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)

def compute_top_order(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
    n, d = X.shape
    order = []
    active_nodes = list(range(d))
    for i in range(d-1):
        H = Stein_hess(X, eta_G, eta_H)
        if normalize_var:
            H = H / H.mean(axis=0)
        if dispersion == "var": # The one mentioned in the paper
            l = int(H.var(axis=0).argmin())
        elif dispersion == "median":
            med = H.median(axis = 0)[0]
            l = int((H - med).abs().mean(axis=0).argmin())
        else:
            raise Exception("Unknown dispersion criterion")
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:,0:l], X[:,l+1:]])
    order.append(active_nodes[0])
    order.reverse()
    return order

def heuristic_kernel_width(X):
    X_diff = X.unsqueeze(1)-X
    D = torch.norm(X_diff, dim=2, p=2)
    s = D.flatten().median()
    return s

def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def heuristic_kernel_width(X):
    X_diff = X.unsqueeze(1)-X
    D = torch.norm(X_diff, dim=2, p=2)
    s = D.flatten().median()
    return s

def Stein_hess_parents(X, s, eta, l): # If one wants to estimate the leaves parents based on the off-diagonal part of the Hessian
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('ikj,ik->ij', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK)
    Gl = torch.einsum('i,ij->ij', G[:,l], G)
    
    nabla2lK = torch.einsum('ik,ikj,ik->ij', X_diff[:,:,l], X_diff, K) / s**4
    nabla2lK[:,l] -= torch.einsum("ik->i", K) / s**2
    
    return -Gl + torch.matmul(torch.inverse(K + eta * torch.eye(n)), nabla2lK)

def Stein_pruning(X, top_order, eta, threshold = 0.1):
    d = X.shape[1]
    remaining_nodes = list(range(d))
    A = np.zeros((d,d))
    for i in range(d-1):
        l = top_order[-(i+1)]
        s = heuristic_kernel_width(X[:, remaining_nodes].detach())
        p = Stein_hess_parents(X[:, remaining_nodes].detach(), s, eta, remaining_nodes.index(l))
        p_mean = p.mean(axis=0).abs()
        s_l = 1 / p_mean[remaining_nodes.index(l)]
        parents = [remaining_nodes[i] for i in torch.where(p_mean > threshold / s_l)[0] if top_order[i] != l]
        #parents = torch.where(p.mean(axis=0) > 0.1)[0]
        A[parents, l] = 1
        A[l, l] = 0
        remaining_nodes.remove(l)
    return A

def SCORE(X, eta_G=0.001, eta_H=0.001, cutoff=0.001, normalize_var=False, dispersion="var", pruning = 'CAM', threshold=0.1):
    top_order = compute_top_order(X, eta_G, eta_H, normalize_var, dispersion)
    return Stein_pruning(X, top_order, eta_G, threshold = threshold), top_order
