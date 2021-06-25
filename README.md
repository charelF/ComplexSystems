# ComplexSystems

A cellular automata


## Files and Folders

- ```img/``` - the visualisations generated for the presentation
- ```data/``` - the data generated from simulations data, like .npy arrays
- ```other/``` - various unrelated files that are relevant to the Complex systems class but not related to the project
- ```code/``` - the code of the project

For the code part, over time we have realised that the best way to work with jupyter notebooks and simulatenously having version control is to have individual folders for each person. People can then put the code they work on in their own folder, which assures that there are not git conflicts since other members are not supposed to work on the files in someone elses folder. Instead they should import or, if not possible, copy them in their own repository. We found this to be the best way to avoid merge conflicts, which are very hard to deal with in jupyter notebooks.

- ```code/shared/wednesdaySPEED``` - the fast Numba implementation of our final algorithm. It is called wednesday as it was created on a wednesday and we were unable to find a better name
- ```code/shared/bartolozziSPEED``` - the fast Numba implementation of the original algorithm by Bartolozzi et al.
- ```code/shared/original_implementation``` - the original implementation of the algorithm by Bartolozzi et al.






| Small | Large |
|------------|------------|
| ![complex](img/CA_small.png) | ![complex](img/CA_large.png) |

Yielding interesting phenomena

| Heat Capacity | Entropy |
|------------|------------|
| ![rotating temperature](img/3DVideo/C_4.gif) | ![entropy](img/3DVideo/S_1.gif) |



## Acknowledgment

Entropy Estimators: https://github.com/gregversteeg/NPEET
