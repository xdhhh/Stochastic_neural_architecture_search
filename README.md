# SNAS(Stochastic Neural Architecture Search)
Pytorch implementation of SNAS (Caution : This is not official version and was not written by the author of the paper)

## Requirements
```
Python >= 3.6.5, PyTorch == 1.0
```

## Datasets
Cifar-10

## Hyperparameters
Overall, I followed hyperparameters that were given in the paper.

However, there are several parameters that were not given in the paper.

Ex) Softmax Temperature ($ \lambda_{0} $) , annealiation rate of the softmax temperature, parameters regarding the levels of resource constraints

Specifically, I found that search validation accuracy is highly influenced by initial softmax temperature
# Run the training code
```
python main_constraint_new.py (WITH resource contraint)

```


## Search Validation Accuracy (with resource constraint)
<p align="center">
<img src="./train.png" alt="train" width="40%">
<img src="./test.png" alt="test" width="40%"></p>
<p align="center">
Figure1 : Search Validation Accuracy
</p>
(Note : the model was not fully trained(<==>converged) due to the limited resources (E.g., GPU and TIME!!)

## Network Architecture (without resource constraint at epoch 70)
<p align="center">
<img src="./Normal_cell.png" alt="Normal Cell" width="40%">
<img src="./Reduction_Cell.png" alt="Reduction Cell" width="40%">
</p>
<p align="center">
Figure2 : Network Architecture of normal cell (left) and reduction cell (right)
</p>

## Network Architecture Comparison (at epoch 30)
<p align="center">
<img src="./Cell_comparison.png"  width="60%">
</p>
<p align="center">

## Reference
https://github.com/quark0/darts/blob/master/README.md
