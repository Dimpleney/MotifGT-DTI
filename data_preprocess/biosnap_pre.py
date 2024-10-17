from random import shuffle

import pandas as pd

from config import Config


def biosnap_process(config: Config):
    print("biosnap_process:")
    # index,sequence,id
    df_pdb = pd.read_csv(config.raw_data_dir + '/BIOSNAP/' + 'full_data/'+ "biosnapSeqpdb.csv")
    # print("df_pdb:",df_pdb)
    df_full = pd.read_csv(config.raw_data_dir + '/BIOSNAP/' + 'full_data/' + "true_1.csv")
    # print("df_full:", df_full)

    # df_full = df_full[:100]
    raw_data_train = df_full[0: int(len(df_full) * config.train_split)]
    print("raw_data_train length:", len(raw_data_train))
    raw_data_valid = df_full[int(len(df_full) * config.train_split): int(
        len(df_full) * (config.train_split + config.val_split))]
    raw_data_test = df_full[int(len(df_full) * (config.train_split + config.val_split)): int(len(df_full))]
    print("raw_data_test length:", len(raw_data_test))
    train_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    for index,row in raw_data_train.iterrows():
        try:
            smiles = row['SMILES']
            # print("smiles:",smiles)
            sequence = row['Target Sequence']
            # print("sequence:",sequence)
            pdb_code = df_pdb.loc[df_pdb["sequence"] == sequence]["pdb_id"].item()
            # print("pdb_code:",pdb_code)
            if pdb_code is not None:
                label = 1 if row['Label'] == 1.0 else 0
                # print("label:",label)
            data_to_append = {'SMILE': smiles, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label}
            train_df = pd.concat([train_df, pd.DataFrame([data_to_append])], ignore_index=True)
        # print("append successful")
        # print("train_df:",train_df)
        except:
            pass

    val_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    for index, row in raw_data_valid.iterrows():
        try:
            smiles = row['SMILES']
            # print("smiles:", smiles)
            sequence = row['Target Sequence']
            # print("sequence:", sequence)
            pdb_code = df_pdb.loc[df_pdb["sequence"] == sequence]["pdb_id"].item()
            # print("pdb_code:", pdb_code)
            if pdb_code is not None:
                label = 1 if row['Label'] == 1.0 else 0
                # print("label:", label)
            data_to_append = {'SMILE': smiles, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label}
            val_df = pd.concat([val_df, pd.DataFrame([data_to_append])], ignore_index=True)
        # print("append successful")
        # print("train_df:",train_df)
        except:
            pass

    test_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    for index, row in raw_data_test.iterrows():
        try:
            smiles = row['SMILES']
            # print("smiles:", smiles)
            sequence = row['Target Sequence']
            # print("sequence:", sequence)
            pdb_code = df_pdb.loc[df_pdb["sequence"] == sequence]["pdb_id"].item()
            # print("pdb_code:", pdb_code)
            if pdb_code is not None:
                label = 1 if row['Label'] == 1.0 else 0
                # print("label:", label)
            data_to_append = {'SMILE': smiles, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label}
            test_df = pd.concat([test_df, pd.DataFrame([data_to_append])], ignore_index=True)
        # print("append successful")
        # print("train_df:",train_df)
        except:
            pass

    train_df_dir = config.df_dir + 'biosnap_train' + '.csv'
    val_df_dir = config.df_dir + 'biosnap_val' + '.csv'
    test_df_dir = config.df_dir + 'biosnap_test' + '.csv'
    train_df.to_csv(train_df_dir)
    val_df.to_csv(val_df_dir)
    test_df.to_csv(test_df_dir)
    return {'train': train_df_dir, 'val': val_df_dir, 'test': test_df_dir}
