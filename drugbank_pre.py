from random import shuffle

import pandas as pd

from config import Config


def drugbank_process(config: Config):
    print("drugbank_process:")
    # index,sequence,id
    df = pd.read_csv(config.raw_data_dir + '/drugbank/' + "drugbankSeqPdb.txt")
    # print("df:",df)
    # smiles sequence label
    with open(config.raw_data_dir + '/drugbank/' + "DrugBank.txt", 'r') as fp:
        train_raw = fp.read()
        # print("train_raw:")
    raw_data = train_raw.split("\n")
    # print("raw_data:",raw_data)
    # print("raw_data length:",len(raw_data))
    shuffle(raw_data)
    # raw_data = raw_data[:100]
    # print("raw_data:",raw_data)
    raw_data_train = raw_data[0: int(len(raw_data) * config.train_split)]
    print("raw_data_train length:", len(raw_data_train))
    # print("raw_data_valid:")
    raw_data_valid = raw_data[int(len(raw_data) * config.train_split): int(
        len(raw_data) * (config.train_split + config.val_split))]
    # print("raw_data_val length:", len(raw_data_valid))
    # print("raw_data_test:")
    raw_data_test = raw_data[int(len(raw_data) * (config.train_split + config.val_split)): int(len(raw_data))]
    print("raw_data_test length:", len(raw_data_test))
    train_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    # print("train_df:", train_df)
    for item in raw_data_train:
        try:
            a = item.split()
            smile = a[2]
            # print(smile)
            sequence = a[3]
            # print("a:",a)
            # PDBID
            pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[0:4]
            # print("pdb_code:", pdb_code)
            if pdb_code is not None:
                # print("pdb_code is not None:")
                label = 1 if a[4] == '1' else 0
                # local
                # train_df = train_df.append({'SMILE': smile, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label},
                #                 ignore_index=True)
                # server
                data_to_append = {'SMILE': smile, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label}
                train_df = pd.concat([train_df, pd.DataFrame([data_to_append])], ignore_index=True)
                # print("append successful")
                # print("train_df:",train_df)
        except:
            pass

    val_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    # print("val_df:", train_df)
    for item in raw_data_valid:
        try:
            a = item.split()
            smile = a[2]
            sequence = a[3]
            pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[0:4]
            if pdb_code is not None:
                label = 1 if a[4] == '1' else 0
                val_df = val_df.append(
                    {'SMILE': smile, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label},
                    ignore_index=True)
        except:
            pass
    # print("test_df:", train_df)
    test_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    for item in raw_data_test:
        try:
            a = item.split()
            smile = a[2]
            sequence = a[3]
            pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[0:4]
            if pdb_code is not None:
                label = 1 if a[4] == '1' else 0
                # test_df = test_df.append(
                #     {'SMILE': smile, 'PDB': pdb_code[0:4], 'TargetSequence': sequence, 'Label': label},
                #     ignore_index=True)
                data_to_append = {'SMILE': smile, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label}
                test_df = pd.concat([test_df, pd.DataFrame([data_to_append])], ignore_index=True)
        except:
            pass
    train_df_dir = config.df_dir + 'drugbank_train' + '.csv'
    val_df_dir = config.df_dir + 'drugbank_val' + '.csv'
    test_df_dir = config.df_dir + 'drugbank_test' + '.csv'
    train_df.to_csv(train_df_dir)
    val_df.to_csv(val_df_dir)
    test_df.to_csv(test_df_dir)
    return {'train': train_df_dir, 'val': val_df_dir, 'test': test_df_dir}

