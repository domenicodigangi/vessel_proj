# Resources on Qquantification of Features' Relevance

## Papers

- https://www.jmlr.org/papers/volume22/20-1316/20-1316.pdf
- https://arxiv.org/pdf/2006.16234.pdf
- https://arxiv.org/pdf/2004.00668.pdf

## Code
- https://github.com/slundberg/shap
- https://github.com/marcotcr/lime
- https://github.com/iancovert/sage/

## Other
- https://towardsdatascience.com/the-4-types-of-additive-feature-importances-5a89f8111996 :
  molto utile per avere un overview

## Comments 
- permutation_importance from sklearn shuffles one feature at a time and checks the average (over repeated shufflings of the same feature) deterioration in performance. 
  - In shuffling takes the independent marginalized resampling approach (does not take features' dependencies into account)
  - Compares the performance of two sets of features 1. full set vs 2 full set - one feature

- Approaches based on Shapley value are theoretically more sound and take into account features' dependencies. SAGE for example:
  - compares all possible subsets of features that do not include feat i with the same subset + i 
  - in removing sets of features it uses a more principled shuffling approach based on conditional marginal distributions