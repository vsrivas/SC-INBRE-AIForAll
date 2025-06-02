# AI for All
Code to reproduce instructional material for AI for All.

## Installation
If a conda session is active run:\
`conda deactivate`

Then run:\
`conda env create -f environment.yml`\
`conda activate aiforall`

## Running Scripts

### Linear regression
`python linear_regression.py --lr <learning_rate> --steps <training_steps> --bias <dataset bias>`\
**Switches**:\
No bias parameter: `--no_bias`

**Examples:**\
No bias parameter: `python linear_regression.py --lr 0.1 --no_bias`\
No bias parameter with biased data: `python linear_regression.py --lr 0.1 --bias 3 --no_bias`
Bias parameter with biased data: `python linear_regression.py --lr 0.1 --bias 3`


### Logistic regression
`python logistic_regression.py --lr <learning_rate> --steps <training_steps>`

**Examples:**\
`python logistic_regression.py`


### MNIST Classification
`python mnist_test.py --nnet <nnet_model> --dim <size_of_plot>`\
**Switches**:\
Linear model: `--lin`

**Examples:**\
Linear model: `python mnist_test.py --nnet models/mnist/mnist_lin.pt --dim 500 --lin`
