# codeme
<ol>
<li><code>git clone https://github.com/acebot712/codeme.git</code>
<li><code>git submodule init</code>
<li><code>git submodule update</code>
<li>Make sure you have environment (conda or pip) with requirements from `requirements.txt`. 
<i>Download torch and tensorflow manually based on GPU availaibility on your machine</i>
    <ol>
    <li>Conda Environment - <code>conda list -e > requirements.txt</code>
    <li>Pip Environment - <code>pip install -r requirements.txt</code>
    </ol>
<li>Activate the environment created above.
</ol>

## Creating datasets
1. Go to `CodeSearchNet/notebooks/ExploreData.ipynb`.
2. The last few lines in the notebook will show how to save `.csv` files for dataset creation.
3. Refer to the other parts of the code to create the necessary variables for the same.

## Running finetuning
1.  Give the correct dataset location `lines 28` and `line 29`
2.  `python3 finetune.py`