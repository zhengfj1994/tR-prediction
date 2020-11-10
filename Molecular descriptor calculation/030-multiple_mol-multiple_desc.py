# Reference code：https://github.com/zhengfj1994/mordred/blob/develop/examples/030-multiple_mol-multiple_desc.py
# Origin author：hirotomo-moriwaki; UnixJunkie; philopon
# Reference：Moriwaki H, Tian Y-S, Kawashita N, Takagi T (2018) Mordred: a molecular descriptor calculator. Journal of Cheminformatics 10:4 . doi: 10.1186/s13321-018-0258-y

# Modified by Zheng Fujian
# Data：2019-6-5

from multiprocessing import freeze_support

from rdkit import Chem

from mordred import Calculator, descriptors

import numpy as np, pandas as pd, os

os.chdir(r'F:/mordred')

f = open("JuRan.txt", "r")
data = f.readlines()
f.close()

if __name__ == "__main__":
    freeze_support()

    mols = [] # 创建list

    count = 1 # 计数用S
    for x in data:
        mols.append(Chem.MolFromSmiles(x))
        print(count)
        print(x)
        count = count+1

    None_index = np.array([x for x in range(len(mols)) if mols[x] == None])
    New_data = np.delete(np.array(data), None_index)
    New_mols = np.delete(np.array(mols), None_index).tolist()
    # Create Calculator
    calc = Calculator(descriptors)

    # map method calculate multiple molecules (return generator)
    # print(list(calc.map(mols)))

    # pandas method calculate multiple molecules (return pandas DataFrame)
    MD_1825 = calc.pandas(New_mols)
    print(MD_1825)
    np.savetxt("New_JuRan.txt", New_data, fmt='%s')
    MD_1825.to_csv('JuRan-SMILES_MDs.csv', index = True, header = True)
