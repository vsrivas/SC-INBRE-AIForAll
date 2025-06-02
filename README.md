# AI for All
Code to reproduce instructional material for AI for All.

## Installation
If a conda session is active run:\
`conda deactivate`

Then run:\
`conda env create -f environment.yml`\
`conda activate aiforall`

## Running Scripts

### Linear models

#### Regression
`python linear_regression.py --lr <learning_rate> --steps <training_steps> --bias <dataset bias>`\
**Switches**:\
No bias parameter: `--no_bias`

**Examples:**\
No bias parameter: `python linear_regression.py --lr 0.1 --no_bias`\
No bias parameter with biased data: `python linear_regression.py --lr 0.1 --bias 3 --no_bias`
Bias parameter with biased data: `python linear_regression.py --lr 0.1 --bias 3`


#### Classification