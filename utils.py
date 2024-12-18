from typing import List, Set, Tuple, Union, Optional

import numpy as np
import logging
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, \
    atom_implicit_valence_one_hot, atom_formal_charge, atom_num_radical_electrons, atom_hybridization_one_hot, \
    atom_is_aromatic, atom_total_num_H_one_hot, ConcatFeaturizer

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import dgl
import networkx as nx
from deepchem.dock import ConvexHullPocketFinder
import macfrag
import torch.nn as nn
import deepchem
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


CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25

Amino_acid_map = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
}

def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


pk = ConvexHullPocketFinder()

def atom_feature(atom):
    return np.array(ConcatFeaturizer([atom_type_one_hot,# 40
                                      atom_degree_one_hot,# 11
                                      atom_implicit_valence_one_hot, # 7
                                      atom_formal_charge,# 1
                                      atom_num_radical_electrons,# 1
                                      atom_hybridization_one_hot,# 5
                                      # 芳香性
                                      atom_is_aromatic,# 1
                                      atom_total_num_H_one_hot])(atom))# 5


def get_atom_feature(m):
    H = []
    for i in range(len(m)):
        H.append(atom_feature(m[i][0]))
    H = np.array(H)

    return H


node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')


def process_protein(pdb_file):

    pdb_id = pdb_file.split('/')[-1][3:-4]
    none_pdb_list = './data/none_pdb_list.txt'
    m = Chem.MolFromPDBFile(pdb_file)
    # print(m)
    if m is None:
        print(f"cannt obtain{pdb_id}的pdb structure")
        with open(none_pdb_list, 'a') as f:
            f.write(pdb_id + '\n')
        return None
    n2 = m.GetNumAtoms()
    if n2 >= 50000:
        print(f"{pdb_id}too much")
        with open(none_pdb_list, 'a') as f:
            f.write(pdb_id + '\n')
        return None

    try:
        am = GetAdjacencyMatrix(m)
    except MemoryError:
        print(f"MemoryError occurred while getting adjacency matrix for {pdb_id}.")
        # file_list.append(pdb_id)
        with open(none_pdb_list, "a") as f:
            f.write(pdb_id + '\n')
        return None
    pockets = pk.find_pockets(pdb_file)
    c2 = m.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    binding_parts = []
    not_in_binding = [i for i in range(0, n2)]
    constructed_graphs = []
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        binding_parts_atoms = []
        idxs = []
        for idx, atom_cord in enumerate(d2):
            if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                idxs.append(idx)
                if idx in not_in_binding:
                    not_in_binding.remove(idx)
        ami = am[np.array(idxs)[:, None], np.array(idxs)]
        H = get_atom_feature(binding_parts_atoms)
        g = nx.convert_matrix.from_numpy_array(ami)
        graph = dgl.from_networkx(g)
        graph.ndata['h'] = torch.Tensor(H)
        graph = dgl.add_self_loop(graph)
        constructed_graphs.append(graph)
        binding_parts.append(binding_parts_atoms)
    constructed_graphs = dgl.batch(constructed_graphs)
    return constructed_graphs



def process_smile_graph(smile, max_block, max_sr, min_frag_atoms):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        frags = macfrag.MacFrag(mol, maxBlocks=max_block, maxSR=max_sr, asMols=False, minFragAtoms=min_frag_atoms)
    else:
        return None

    g = [dgl.add_self_loop(smiles_to_bigraph(fr, node_featurizer=node_featurizer)) for fr in frags]
    g = dgl.batch(g)

    return g

# for pro_sequence
def integer_label_sequence(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of the input protein string.
    """
    encoding = np.zeros(max_length, dtype=int)  # Change dtype to int
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exist in sequence category encoding, skip and treat as " f"padding."
            )
    return torch.tensor(encoding)  # Convert to torch.tensor with int dtype


class ProteinEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(ProteinEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, sequence):
        """
        Args:
            sequence (torch.Tensor): Integer encoded amino acid sequence

        Returns:
            torch.Tensor: A sequence of embedded representations.
        """
        embedded_sequence = self.embedding_layer(sequence)
        embedded_sequence = embedded_sequence.transpose(0, 1)
        return embedded_sequence


def embed_protein_sequence(sequence, embedding_dim=32):
    """
    Embeds a protein sequence by first integer encoding and then using an embedding layer.

    Args:
        sequence (str): Protein string sequence.
        embedding_dim (int): Dimension of the embedding.

    Returns:
        torch.Tensor: Embedded representation of the protein sequence.
    """
    encoded_sequence = integer_label_sequence(sequence)
    emb_model = ProteinEmbedding(len(CHARPROTSET) + 1, embedding_dim)
    embedded_sequence = emb_model(encoded_sequence)
    return embedded_sequence

def one_hot_encode(sequence, amino_acid_map=Amino_acid_map, max_length=1200):
    """
    The sequence length is fixed to max_length
    """
    num_classes = len(amino_acid_map)
    sequence_length = len(sequence)
    if sequence_length < max_length:

        one_hot = torch.zeros((num_classes, max_length), dtype=torch.float32)
        for i, amino_acid in enumerate(sequence):
            if amino_acid in amino_acid_map:
                one_hot[amino_acid_map[amino_acid], i] = 1
    else:
        one_hot = torch.zeros((num_classes, max_length), dtype=torch.float32)
        for i, amino_acid in enumerate(sequence[:max_length]):
            if amino_acid in amino_acid_map:
                one_hot[amino_acid_map[amino_acid], i] = 1
    return one_hot


# test
# pdb_file = "./data/pdb/pdb4l1d.pdb"
# dgl = Chem.MolFromPDBFile(pdb_file)
# print(dgl)