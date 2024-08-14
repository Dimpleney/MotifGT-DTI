'''
Protein features are preprocessed to improve training efficiency
'''

import os
import dgl
from protein_process import prot_to_graph
from utils import process_protein

def protein_pre():
    pdb_directory = './data/pdb'
    graph_directory = './data/features'

    if not os.path.exists(graph_directory):
        os.makedirs(graph_directory)

    for filename in os.listdir(pdb_directory):
        if filename.endswith('.pdb'):
            pdb_file_path = os.path.join(pdb_directory, filename)
            pdbid = filename.split('.')[0]
            print("processing：",pdbid)
            graph_data_file = os.path.join(graph_directory, f'{pdbid}.bin')

            if not os.path.exists(graph_data_file):
                graph = prot_to_graph(pdb_file_path)
                if graph is None:
                    print(f"Failed to create a graph object from {pdb_file_path}, skipping this file.")
                    continue

                dgl.save_graphs(graph_data_file, [graph])

                print(f"Saved DGL graph data for {pdbid} to {graph_data_file}")
            else:
                print(f"Graph data for {pdbid} already exists, skipping.")

    print("Finished processing and saving all pdb files as DGL graph objects.")



protein_pre()