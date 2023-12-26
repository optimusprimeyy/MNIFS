# fuzzy Multi-Neighborhood entropy-based Interactive unsupervised Feature Selection (MNIFS) algorithm

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
from scipy.io import loadmat


def Fmg_entropy(granule):
    # input:
    # granule is fuzzy multi-neighborhood granule for attributes.
    entropy = sum(-(1 / granule.shape[0]) * (np.log2(1 / (granule.sum(axis=0)))))
    return entropy


def relation_matrix(Data, delta):
    # input:
    # Data is data matrix without decisions, where rows for samples and columns for attributes.
    # Numerical attributes should be normalized into [0,1].
    # Nominal attributes be replaced by different integer values.
    # delta is a given parameter for calculating the fuzzy similarity relation.
    datatrans = np.zeros((len(Data), 2))
    datatrans[:, 0] = Data
    temp = pdist(datatrans, 'cityblock')  # Calculate Manhattan Distance
    temp = squareform(temp)  # Convert the resulting one-dimensional matrix into a two-dimensional matrix
    temp[temp > delta] = 1
    R = 1 - temp
    return R


def MNIFS(Data, lammda):
    # input:
    # Data is data matrix without decisions, where rows for samples and columns for attributes.
    # Numerical attributes should be normalized into [0,1].
    # Nominal attributes be replaced by different integer values.
    # lammda is the adjustment parameter for the neighborhood radius.

    row, attrinu = Data.shape  # Get the number of rows and columns of data, row: number of objects, attrinu: number of attributes

    delta = np.zeros(attrinu)  # Multi-neighborhood radius for attributes

    ID = (Data <= 1).all(axis=0)
    delta[ID] = (np.std(Data[:, ID], axis=0)) / lammda

    Select_Fea = []  # The set of features that have been selected
    sig = []  # Used to record the value of the indicator when each feature is selected

    E = np.zeros(attrinu)  # Initialize fuzzy multi-granularity entropy
    Joint_E = np.zeros((attrinu, attrinu))  # Initialize fuzzy multi-granularity joint entropy
    MI = np.zeros((attrinu, attrinu))  # Initialize fuzzy mutual information

    # 1.1 Calculate fuzzy multi-granularity entropy
    for j in range(0, attrinu):
        r = relation_matrix(Data[:, j], delta[j])  # fuzzy multi-neighborhood granule for attribute j.
        E[j] = Fmg_entropy(r)

    # 1.2 Computing fuzzy multi-granularity joint entropy and fuzzy mutual information
    for i in range(0, attrinu):
        ri = relation_matrix(Data[:, i], delta[i])  # Fuzzy relation matrix corresponding to the i-th attribute
        for j in range(0, i + 1):
            rj = relation_matrix(Data[:, j], delta[j])  # # Fuzzy relation matrix corresponding to the j-th attribute
            Joint_E[i, j] = Fmg_entropy(
                np.minimum(ri, rj))  # Calculate the joint entropy of attribute i and attribute j
            Joint_E[j, i] = Joint_E[i, j]

            # Calculate the fuzzy mutual information of attribute i and attribute j
            MI[i, j] = E[i] + E[j] - Joint_E[i, j]
            MI[j, i] = MI[i, j]

    # 1.3 Calculate the fuzzy correlation, i.e., the average of the fuzzy mutual information
    Ave_MI = np.mean(MI, axis=1)
    n1 = (np.argsort(Ave_MI)[::-1]).tolist()  # Sorted index values
    Ave_MI_Sorted = (Ave_MI[np.argsort(Ave_MI)[::-1]]).tolist()  # Sorted Ave_MI

    sig.append(Ave_MI_Sorted[0])
    Select_Fea.append(n1[0])
    unSelect_Fea = n1[1:]

    # 2. Calculate the redundancy and interactivity of unselected features
    while unSelect_Fea:
        Red = np.zeros((len(unSelect_Fea), len(Select_Fea)))  # Initialize redundancy
        # i is an unselected feature
        for i in range(0, len(unSelect_Fea)):
            # j is selected feature
            for j in range(0, len(Select_Fea)):
                FE = Joint_E[Select_Fea[j], unSelect_Fea[i]]
                Red[i, j] = Ave_MI[Select_Fea[j]] - FE / E[Select_Fea[j]] * Ave_MI[Select_Fea[j]]

        # 2.1 Calculate the average value of redundancy corresponding to each unselected feature
        Ave_FRed = np.mean(Red, axis=1)

        # 2.2 Calculate Interactivity
        Itr = np.zeros((len(unSelect_Fea), len(unSelect_Fea)))

        # If there is only one unselected feature left, the interactivity is 0
        if len(unSelect_Fea) == 1:
            Ave_Itr = np.sum(Itr, axis=1)

        else:  # Calculate the relation matrix using the intersection method
            srrcj = np.ones((row, row))
            for j in range(0, len(Select_Fea)):
                srr_Select_j = relation_matrix(Data[:, Select_Fea[j]], delta[Select_Fea[j]])
                srrcj = np.minimum(srrcj, srr_Select_j)

            # Iterate over all unselected features and compute the interaction of the current candidate feature c
            for c in range(0, len(unSelect_Fea)):
                for i in range(0, len(unSelect_Fea)):  # Iterate over all unselected features
                    if c == i:
                        continue
                    srr_UnSe_i = relation_matrix(Data[:, unSelect_Fea[i]], delta[unSelect_Fea[i]])
                    srr_UnSe_c = relation_matrix(Data[:, unSelect_Fea[c]], delta[unSelect_Fea[c]])
                    Joint_Three = Fmg_entropy(np.minimum(np.minimum(srr_UnSe_i, srrcj), srr_UnSe_c))
                    Joint_Two = Fmg_entropy(np.minimum(srrcj, srr_UnSe_c))
                    Itr[c, i] = Joint_E[unSelect_Fea[i], unSelect_Fea[c]] + Joint_Two - Joint_Three - E[unSelect_Fea[c]]
                    Itr[c, i] = np.abs(Itr[c, i])

            # Compute average interactivity.
            Ave_Itr = np.sum(Itr, axis=1)
            Ave_Itr = Ave_Itr / (len(unSelect_Fea) - 1)

        # 3. Calculate maximum correlation minimum redundancy maximum interaction
        UFmRMR = Ave_MI[unSelect_Fea] - Ave_FRed + Ave_Itr
        UFmRMR = UFmRMR.tolist()

        # Select the attribute that satisfies the maximum correlation minimum redundancy maximum interaction
        max_sig = max(UFmRMR)
        max_tem = UFmRMR.index(max(UFmRMR))
        sig.append(max_sig)
        Select_Fea.append(unSelect_Fea[max_tem])  # Add to the ordered sequence of features
        unSelect_Fea.pop(max_tem)  # Remove from candidate feature set

    return Select_Fea, sig


if __name__ == "__main__":
    # Import data
    Example = loadmat('Example.mat')
    trandata = np.array(Example['Example'])

    # Normalization process
    min_max_scaler = preprocessing.MinMaxScaler()
    trandata[:, 0:2] = min_max_scaler.fit_transform(trandata[:, 0:2])

    # Set the threshold lammda
    lammda = 1

    feature_seq, sig = MNIFS(trandata, lammda)
    print(feature_seq)
