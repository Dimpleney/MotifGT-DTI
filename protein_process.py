
import os
import pickle
import timeit
import numpy as np
import pandas as pd
import torch
import dgl
from deepchem.dock import ConvexHullPocketFinder
from rdkit import Chem
from scipy import sparse as sp
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs
import warnings

warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Metallic elements
METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
# Maximum number of atoms of a residue
RES_MAX_NATOMS = 24

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# Calculate the distance information of the atoms
def obtain_self_dist(res):
    try:
        # xx = res.atoms.select_atoms("not name H*")
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        # 1 angstrom =0.1 nanometers
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]

#Dihedral Angle: parameter to describe the conformation of main and side chains of proteins
def obtain_dihediral_angles(res):
    try:
        # Phi（φ）
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        # Psi（ψ）
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        # Omega（ω）
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        # Chi1（χ1）
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]

# initial embedding：41-dimension
def calc_res_features(res):
    # one-hot: 20 amino acids, distance coding, dihedral Angle coding
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32  residue type
					obtain_self_dist(res) +
					obtain_dihediral_angles(res)
					)

# Classify the residues
def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname


##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',
# if less than the truncation value then exists edge
def obatin_edge(u, cutoff=10.0):
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T

# check the connection between resiudes
def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0

def calc_dist(res1, res2):

    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array

def load_protein(protpath, explicit_H=False, use_chirality=True):

    mol = Chem.MolFromPDBFile(protpath, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def obtain_ca_pos(res):
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:  ##some residues loss the CA atoms
            return res.atoms.positions.mean(axis=0)


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    # If your version of DGL requires specifying the sparse format like this, uncomment the following line:
    # A = g.adjacency_matrix().astype(float)  # DGL version may vary

    # For newer versions of DGL, you should directly use adjacency_matrix() like this:
    A = g.adjacency_matrix().to_dense().numpy().astype(float)

    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Compute the eigenvalues and eigenvectors
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    if EigVec.shape[1] < pos_enc_dim + 1:
        # If not enough eigenvectors, pad with zeros
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)

    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g

class ProteinTooBig(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, size, pdb, message="Protein size is too big to parse"):
        self.size = size
        self.pdb = pdb
        self.message = message
        super().__init__(self.message + f" {pdb} size is {str(size)}")

def prot_to_graph(prot_pdb, cutoff=10.0):
    """obtain the residue graphs"""
    # pocket object
    pk = ConvexHullPocketFinder()
    # Load a protein structure
    prot = Chem.MolFromPDBFile(prot_pdb, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)

    if prot is None:
        print(f"Failed to load protein from {prot_pdb}.")
        return None
    n2 = prot.GetNumAtoms()
    if n2 >= 50000:
        return None
    Chem.AssignStereochemistryFrom3D(prot)
    u = mda.Universe(prot)
    # dgl object
    g = dgl.DGLGraph()

    num_residues = len(u.residues)
    g.add_nodes(num_residues)
    # Initialize the feature vector for each amino acid
    res_feats = np.array([calc_res_features(res) for res in u.residues])
    g.ndata["h"] = torch.tensor(res_feats,dtype=torch.float32)
    # Calculate the edge and maximum distance according to cutoff
    edgeids, distm = obatin_edge(u, cutoff)
    src_list, dst_list = zip(*edgeids)
    g.add_edges(src_list, dst_list)
    # Calculate the coordinates of each residue Cα
    g.ndata["ca_pos"] = torch.tensor(np.array([obtain_ca_pos(res) for res in u.residues]),dtype=torch.float32)
    # Centroid coordinates
    g.ndata["center_pos"] = torch.tensor(u.atoms.center_of_mass(compound='residues'),dtype=torch.float32)

    dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
    cadist = torch.tensor([dis_matx_ca[i, j] for i, j in edgeids],dtype=torch.float32) * 0.1
    dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
    cedist = torch.tensor([dis_matx_center[i, j] for i, j in edgeids],dtype=torch.float32) * 0.1
    edge_connect = torch.tensor(np.array([check_connect(u, x, y) for x, y in zip(src_list, dst_list)]),dtype=torch.float32)

    g.edata["e"] = torch.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), torch.tensor(distm,dtype=torch.float32)], dim=1)
    g.ndata.pop("ca_pos")
    g.ndata.pop("center_pos")

    ca_pos = np.array(np.array([obtain_ca_pos(res) for res in u.residues]))
    pockets = pk.find_pockets(prot_pdb)
    # print(pockets)
    # box corrdinates
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        # node index in pockets
        idxs = []

        for idx in range(ca_pos.shape[0]):
            if x_min < ca_pos[idx][0] < x_max and y_min < ca_pos[idx][1] < y_max and z_min < ca_pos[idx][2] < z_max:
                idxs.append(idx)
    g_pocket = dgl.node_subgraph(g, idxs)
    g_pocket = laplacian_positional_encoding(g_pocket, pos_enc_dim=8)
    return g_pocket


# test_1
# pdb = './data/pdb/pdb1fgx.pdb'
# pk = ConvexHullPocketFinder()
# pockets = pk.find_pockets(pdb)
# print(pockets)

