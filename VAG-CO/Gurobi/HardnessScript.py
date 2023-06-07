import numpy as np
from scipy import linalg


def get_eigenvector(senders, receivers, edges, N):

    coupling_mat = np.zeros((N, N))
    coupling_mat[senders, receivers] = edges[:,0]

    w, v = linalg.eigh(coupling_mat, subset_by_index = [0 ,2])

    test = np.tensordot(coupling_mat, v[:,0], axes=([1],[0]))
    ev_check = np.tensordot(v[:,0], test, axes = ([0],[0]))

    if(not np.allclose(coupling_mat, coupling_mat.transpose())):
        ValueError("Matrix is not Symmetric")
    else:
        print("Matrix is symmetric")
    if(not np.allclose(w, ev_check)):
        ValueError("eigenvalue check indicates that something went wrong!")
    else:
        print("Eigencevtor check is correct")

    return w[0], v[: ,0]

def get_eigenvector_2(senders, receivers, edges, N):

    # Define a matri
    coupling_mat = np.zeros((N, N))
    coupling_mat[senders, receivers] = edges[:,0]

    print(coupling_mat)

    # Get the eigenvalues and eigenvectors of the matrix
    eigenvalues, eigenvectors = np.linalg.eig(coupling_mat)

    # Find the index of the smallest eigenvalue
    min_index = eigenvalues.argmin()

    # Get the corresponding eigenvector
    smallest_eigenvector = eigenvectors[:, min_index]
    smallest_eigenvalue = eigenvalues[min_index]
    return smallest_eigenvalue, smallest_eigenvector


def compute_hardness(cont_gs, gs_spin, spin_flip_symmetry = False, spin_input = True):
    if(spin_input):
        norm = 2
    else:
        norm = 1

    if(spin_flip_symmetry):
        dist = min([np.mean(np.abs(sign(cont_gs) - gs_spin)), np.mean(np.abs(sign(cont_gs) - -gs_spin))])
        dist /= norm
    else:
        dist = np.mean(np.abs(sign(cont_gs) - gs_spin))
    hardness = dist
    return hardness

def compute_relax_Energy_factor(vec):
    factor = 1/np.max(vec)
    return factor

def sign(x):

    th_x = 1.*(x > 0.)
    sign_x = 2*th_x -1
    return sign_x
