# adaboost
basic adaboost 를 decision tree 기반으로 구현하고,
A new diverse adaboost classifier 와 Robust Alternating AdaBoost 논문을 구현.

기존의 AdaBoost의 weak learner 는 Decision tree classifier(max_depth=1)를 사용.

## data)
UCI 에 게재된 data set을 사용.

1. Diabetes
2. Indian Liver Diseases
3. Ionosphere

## code

* Adaboost_algo.py
 including baseline Adaboost, RAdaboost, Dadaboost.
 In case of Dadaboost, In reference(1), diverse AdaBoost is maded based Gentle Adaboost, but I made DadaBoost based Adaboost for comparing Adaboost and RadaBoost, DadaBoost.

* the awesome baseline code is in in #### adaboost.py using reference(3). 



## reference)
(1) An, T. K., & Kim, M. H. (2010, October). A new diverse AdaBoost classifier. In 2010 International Conference on Artificial Intelligence and Computational Intelligence (Vol. 1, pp. 359-363). IEEE.

(2) Allende-Cid, H., Salas, R., Allende, H., & Nanculef, R. (2007, November). Robust alternating adaboost. In Iberoamerican Congress on Pattern Recognition (pp. 427-436). Springer, Berlin, Heidelberg.

(3) https://geoffruddock.com/adaboost-from-scratch-in-python/
