import os
from random import shuffle

import dgl
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import process_protein, process_smile_graph, integer_label_protein,one_hot_encode
from tqdm import tqdm
from Bio.PDB import PDBList
import pickle
import subprocess
from signal import signal, SIGSEGV
from compound_process import smiles_to_graph
from protein_process import prot_to_graph

class DTIData(Dataset):
    # print("class DTIData：")
    def __init__(self, name, df_dir, processed_file_dir, pdb_dir, p_graph, s_graph):
        super().__init__()
        print("class DTIData init：")
        self.p_graph = p_graph
        self.s_graph = s_graph
        self.name = name
        self.df_dir = df_dir
        # print("Init df_dir:", self.df_dir)

        self.df = pd.read_csv(df_dir,header=0)
        print("Init df:",self.df.head())
        self.pdb_dir = pdb_dir
        self.processed_file_dir = processed_file_dir + self.name + '.pkl'
        # create processed.pkl
        if not os.path.exists(processed_file_dir):
            os.mkdir(processed_file_dir)
            print("build processed.pkl")

        if not os.path.exists(self.processed_file_dir):
            print(f"Processed file {self.processed_file_dir} not found. Running pre-processing.")
            self.p_graph = {}
            self.s_graph = {}
            print("pre_process:")
            self.pre_process()
        else:
            print("Processed file found. Loading data.")
            self.p_graph = {}
            self.s_graph = {}
            self.pre_process()
            self.df = self.df[self.df['PDB'].isin(self.p_graph.keys())]
            self.df = self.df[self.df['SMILE'].isin(self.s_graph.keys())]


    def pre_process(self):
        # print("class DTIData pre_process:")
        not_available = []
        not_available_pdb = []
        print("start class DTIData pre_process:")
        # print("i",self.df.iloc[1]['SMILE'])

        for i in tqdm(range(len(self.df.index))):
            # print("loop:")
            smile = self.df.iloc[i]['SMILE']
            # print("smile:", smile)
            # pdb = self.df.iloc[i]['PDB'].lower()
            pdb = self.df.iloc[i]['PDB']
            if not pd.isna(pdb):
                pdb = pdb.lower()
            # print("pdb:", pdb)
            if pdb not in self.p_graph.keys():
                if pdb in not_available_pdb:
                    not_available.append(i)
                    continue
                try:
                    pdb_file_path = os.path.join(self.pdb_dir, f"pdb{pdb}.pdb")
                    # pdb_file_path = os.path.join(self.pdb_dir, f"{pdb}.pdb")
                    if not os.path.exists(self.pdb_dir + "pdb" + pdb + '.pdb'):
                        pdbl = PDBList(verbose=False)
                        pdbl.retrieve_pdb_file(
                            pdb, pdir=self.pdb_dir, overwrite=False, file_format="pdb"
                        )
                        # Rename file to .pdb from .ent
                        os.rename(
                            # self.pdb_dir + "pdb" + pdb + ".ent", self.pdb_dir + pdb + ".pdb"
                            self.pdb_dir + "pdb" + pdb + ".ent", self.pdb_dir + "pdb" + pdb + ".pdb"
                        )


                        # Assert file has been downloaded
                        assert any(pdb in s for s in os.listdir(self.pdb_dir))
                    # constructed_graphs = process_protein(pdb_file_path)
                    features_dir = './data/features/'
                    feature_file = os.path.join(features_dir, f'pdb{pdb}.bin')
                    print("pdb id",pdb)
                    print(feature_file)
                    if os.path.isfile(feature_file):

                        graphs, _ = dgl.load_graphs(feature_file)
                        constructed_graphs = graphs[0]
                    else:
                        print('constructing')
                        constructed_graphs = prot_to_graph(pdb_file_path)
                    if constructed_graphs is not None:
                        print(constructed_graphs)
                        self.p_graph[pdb] = constructed_graphs

                except Exception as e:
                    not_available_pdb.append(pdb)
                    not_available.append(i)

            if smile not in self.s_graph:
                try:
                    self.s_graph[smile] = smiles_to_graph(smile,6,8,1)
                    print("smile:",self.s_graph[smile])
                except Exception as e:
                    print(e)
                    not_available.append(i)

        self.df.drop(list(set(not_available)), axis=0, inplace=True)
        self.df.to_csv(self.df_dir)
        with open(self.processed_file_dir, 'wb') as fp:
            pickle.dump([self.p_graph, self.s_graph], fp)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        # add sequence
        smile = self.df.iloc[index]['SMILE']
        pdb = self.df.iloc[index]['PDB'].lower()
        pdb_sequence = self.df.iloc[index]['TargetSequence']
        p_graph = self.p_graph[pdb]
        s_graph = self.s_graph[smile]
        p_sequence = one_hot_encode(pdb_sequence)
        y = torch.tensor(self.df.iloc[index]['Label'])
        return p_graph, s_graph,p_sequence, y



def collate_wrapper(batch):
    # batch = [
    #     (prot_graph_1, target_graph_1, label_1),
    #     (prot_graph_2, target_graph_2, label_2),
    #     ...
    #     (prot_graph_n, target_graph_n, label_n)
    # ]
    transposed_data = list(zip(*batch))
    # transposed_data = [
    #     (prot_graph_1, prot_graph_2, ..., prot_graph_n),
    #     (target_graph_1, target_graph_2, ..., target_graph_n),
    #     (label_1, label_2, ..., label_n)
    # ]
    prot_graph = transposed_data[0]
    # prot_graph = (prot_graph_1, prot_graph_2, ..., prot_graph_n)
    target_graph = transposed_data[1]
    # target_graph = (target_graph_1, target_graph_2, ..., target_graph_n)
    pdb_sequences = torch.stack(transposed_data[2],dim=0)
    # pdb_sequences = torch.tensor(transposed_data[2], dtype=torch.float32)
    inp = (prot_graph, target_graph, pdb_sequences)
    # inp = ((prot_graph_1, prot_graph_2, ..., prot_graph_n), (target_graph_1, target_graph_2, ..., target_graph_n))
    tgt = torch.stack(transposed_data[3], 0)
    # tgt = torch.tensor(transposed_data[3], dtype=torch.float32)
    # tgt = torch.stack((label_1, label_2, ..., label_n), 0)
    return inp, tgt


