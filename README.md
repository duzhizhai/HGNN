# HGNN
This repository is an implementation of our proposed HGNN(Hierarchical Graph Neural Networks) model in the following paper:

``Identifying User Geolocation with Hierarchical Graph Neural Networks and Explainable Fusion``

# Requirements
python 3.7

torch 1.1.0

tensorflow 1.13.1

check requirements.txt for more details.

# File Structure

- data/cmu/ : the original data of CMU (i.e., GeoText).
- result/model/ : the save path of trained model.
- influence/plot_functions : plot many pics about influence functions.
- influence/get_influ_matrix.py : get influence matrix (shape: (#train, #test)) from ./influence/Res_inf_HGNN/
- influence/Res_inf_HGNN/ : save the influence value of every test samples.
- influence/main_HGNN_inf.py : apply influence functions on HGNN model.
- content_by_d2v.py : get and save doc2vec embedding of tweets contents.
- dataProcess.py : process data and save the processed features into files.
- geoMain.py : the main function.
- kdtree.py : improved kd-tree algorithms to control the bucket size.
- models.py : the implemented models.
- plotFunc.py : the code of plot embedding.
- requirements.txt : the main requirement environment list.

