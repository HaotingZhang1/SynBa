import pandas as pd
import numpy as np


data = pd.read_csv('data/nci_almanac/NCI-ALMANAC_subset_555300.csv', na_values=['.', 'ND'])
data = data.to_numpy()
drugs = np.unique(data[:, 2])
cell_lines = np.unique(data[:, 4])
D = len(drugs)
C = len(cell_lines)
assert D == 50
assert C == 60
l = np.array([])
for d1 in range(D):
    d1_name = drugs[d1].replace(" ", "-")
    ind_1 = np.where(data[:, 2] == drugs[d1])[0]
    ind_2 = np.where(data[:, 3] == drugs[d1])[0]
    data_1 = data[ind_1, :]
    data_2 = data[ind_2, :]
    for d2 in np.arange(start=d1+1, stop=D):
        d2_name = drugs[d2].replace(" ", "-")
        ind_3 = np.where(data_1[:, 3] == drugs[d2])[0]
        ind_4 = np.where(data_2[:, 2] == drugs[d2])[0]
        data_3 = data_1[ind_3, :]
        data_4 = data_2[ind_4, :]
        data_5 = np.copy(data_4)
        data_5[:, 0] = data_4[:, 1]
        data_5[:, 1] = data_4[:, 0]
        data_5[:, 2] = data_4[:, 3]
        data_5[:, 3] = data_4[:, 2]
        data_6 = np.concatenate((data_3, data_5), axis=0)
        for c in range(C):
            c_name = cell_lines[c]
            c_name = c_name.replace(" ", "-")
            c_name = c_name.replace("/", "-")
            ind_5 = np.where(data_6[:, 4] == cell_lines[c])[0]
            data_7 = data_6[ind_5, :]
            if len(ind_5) > 0:
                l = np.append(l, len(ind_5))
                np.save('data/nci_almanac/combinations/' + d1_name + '.' + d2_name + '.' + c_name + '.npy', data_7)

unique, counts = np.unique(l, return_counts=True)
print(dict(zip(unique, counts)))   # Output: {15.0: 35220, 30.0: 900}
