# ALKIA-X
ALKIA-X, the **A**daptive and **L**ocalized **K**ernel **I**nterpolation **A**lgorithm with e**X**trapolated reproducing kernel Hilbert space norm, is an algorithm to automatically approximate black-box functions. 
ALKIA-X ensures:
* A fast-to-evaluate approximating function;
* The guaranteed satisfaction of the desired bound on the approximation error;
* Well-conditioned computations;
* High level of parallelization.
We highlight the benefits of ALKIA-X by automatically approximating nonlinear model predictive control (MPC) schemes with closed-loop guarantees. 


## How to cite ALKIA-X
@article{Tokmak2023Arxiv, \
  title={Automatic nonlinear {MPC} approximation with closed-loop guarantees}, \
  author={Tokmak, Abdullah and Fiedler, Christian and Zeilinger, Melanie N. and Trimpe, Sebastian and Köhler, Johannes}, \
  journal={arXiv preprint arXiv:2312.10199v1}, \
  year={2023} \
}



## Installation guide and requirements
Create a conda environment 
```
conda create --name ALKIAX
```
and install the following packages:
```
pip install scikit-learn
pip install casadi
pip install tqdm
```


## Reproducing experiments
To reproduce the approximation of the MPC scheme for the continious stirred tank reactor, use the following hyperparameters in the ALKIAX_main.py file:
```python
epsilon = 5.1e-3
round_n_digits = 14  # rounding
gt_string = 'CSTR_python'
x_dim, y_dim = ground_truth_dimensions(gt_string)
number_of_head_nodes = 5**2
p_min = 2
cond_max = 1.2e8
C_ell = 0.8
kernel = matern_kernel(sigma=1, ell=C_ell, nu=3/2)
parallel = True
max_storage_termination = np.infty
```

To reproduce the approximation of the MPC scheme for the cold atmospheric plasma device, use the following hyperparameters in the ALKIAX_main.py file:
```python
epsilon = 1e-6
round_n_digits = 14  # rounding
gt_string = 'plasma_python'
x_dim, y_dim = ground_truth_dimensions(gt_string)
number_of_head_nodes = 3**3
p_min = 2
cond_max = 3.4e7
C_ell = 0.8
kernel = matern_kernel(sigma=1, ell=C_ell, nu=3/2)
parallel = True
max_storage_termination = 75
```

Then, run ALKIA-X as follows:
You can run ALKIA-X as follows:
```
conda activate ALKIAX
taskset -c 1,2,3,4,5,6,7,8 python ALKIAX_main.py
```
This exact line will lead to a parallelization on CPU cores 1-8.

## Using pre-trained approximations 
The pre-trained pickle files for the continous stirred tank reactor or the cold atmospheric plasma device are "CSTR_final.pickle" and "plasma_final.pickle", respectively.


## Post-processing





