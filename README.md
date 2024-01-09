# MNIFS
Sihan Wang<sup>1</sup>, Siyu Yang<sup>1</sup>, **Zhong Yuan***, Hongmei Chen, and Dezhong Peng, [Fuzzy multi-neighborhood entropy-based interactive feature selection for unsupervised outlier detection](Paper/2024-MNIFS.pdf), Applied Soft Computing. (Code)

## Abstract
Unsupervised feature selection is one of the important techniques for unsupervised knowledge discovery, which aims to reduce the dimensionality of conditional feature sets as much as possible to improve the efficiency and accuracy of the algorithm. However, existing methods have the following two challenges: 1) They are mainly applicable to select numerical or nominal features and cannot effectively select heterogeneous features; 2) The relevance and redundancy are primarily considered to construct feature evaluation indexes, ignoring the interaction information of heterogeneous features. To solve the challenges mentioned above, this paper proposes an unsupervised heterogeneous feature selection method based on fuzzy multi-neighborhood entropy, which also considers the multi-correlation of features to select heterogeneous features. First, the fuzzy multi-neighborhood granule is constructed by considering the distribution characteristics of the data. Then, the concept of fuzzy entropy is introduced to define the fuzzy multi-neighborhood entropy and its associated uncertainty measures, and the relationship between them is discussed. Next, the relevance, redundancy, and interactivity among attributes are defined, and the idea of maximum relevance-minimum redundancy-maximum interactivity is used to construct the evaluation indexes of heterogeneous features. Finally, experiments are conducted on several publicly unbalanced datasets, and the results are in comparison with existing algorithms. The experimental results show that the proposed algorithm is able to select fewer heterogeneous features to improve the efficiency of outlier detection tasks. The code is publicly available online at [https://github.com/optimusprimeyy/MNIFS](https://github.com/optimusprimeyy/MNIFS).
## Usage
You can run MNIFS.py:
```
if __name__ == "__main__":
     # Import data
    Example = loadmat('Example.mat')
    trandata = np.array(Example['Example'])

    # Normalization process for nominal data
    min_max_scaler = preprocessing.MinMaxScaler()
    trandata[:, 2:4] = min_max_scaler.fit_transform(trandata[:, 2:4])

    # Set the threshold lammda
    lammda = 1

    feature_seq, sig = MNIFS(trandata, lammda)
    print(feature_seq)
```
You can get outputs as follows:
```
feature_seq = [2, 0, 3, 1]
```

## Citation
If you find MNIFS useful in your research, please consider citing:
```

```
## Contact
If you have any question, please contact [wangsihan0713@foxmail.com](wangsihan0713@foxmail.com) or [yangsy0224@foxmail.com](yangsy0224@foxmail.com).

