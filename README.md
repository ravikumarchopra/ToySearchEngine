# Toy Search Engine

## Introduction

We are using the same template initially provided for the Assignments for this project. We have added one more command-line argument named `-model` to `main.py` file which takes the name of model we want use for our Information Retrieval System. And the valid values for argument `-model` are following:

* `VSM (default)` : If we Pass this as a value for argument `-model` then the *Vector Space Model* will be used for retrieving relevant documents for a given query by our Information Retrieval System.

* `LSA` : If we Pass this as a value for argument `-model` then the *Latent Semantic Analysis* will be used for retrieving relevant documents for a given query by our Information Retrieval System.

* `ESA` : If we Pass this as a value for argument `-model` then the *Explicit Semantic Analysis* will be used for retrieving relevant documents for a given query by our Information Retrieval System.

The articles we fetched from wikipedia are stored in a file named `articles.json` inside the folder named `wikipedia`.

## Modules/Packages Used

* `numpy` :  Used for mathematical operations such as matrix multiplication, dot product of vectors for computing cosine similarity and SVD etc.

* `wikipedia` : Used for fetching wikipedia articles for words in document title and body.

* `json` : Used for reading and writing documents in json format.

* `matplotlib` : Used for plotting the performance measure values i.e. *Precision*, *Recall*, *F-Score*, *MAP* and *nDCG*.

* `tqdm` : Used for showing the progress and time it took while executing.

## To execute and verify the code please follow the following steps:

* Check that all the requirements with their specific versions in the `requirements.txt` file are satisfied.
* Download and unzip the code folder.
* Open the folder in cmd.
* You can run the project by executing the file named `main.py`.
* Pass the name (e.g. `VSM|LSA|ESA`) of the model you want to use in argument `-model`.
* You can see the value for *Precision*, *Recall*, *MAP* and *nDCG* in the output.
* You can also see the plot for *Precision*, *Recall*, *MAP* and *nDCG* in the file inside `output` folder named as `eval_plot.png`.