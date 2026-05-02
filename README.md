# COMP597 Computer Vision Final Project - Spring 2026

## Penn State University Harrisburg

Ryan Chang, Joel Horne, Haytham Zaami

The `COMP597_FINAL.ipynb` Jupyter Notebook contains the training and evaluation code and loss results for our SimCLR and ProxyNCA implementations.

`cub2011.py` is a helper script that defines a torch.utils.data.DataSet class. It is adapted from https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/cub2011.py. The script pulls the CUB-2011-200 dataset from https://www.vision.caltech.edu/datasets/cub_200_2011/, extracts the raw images and labels, and organizes it as a DataSet class similar to PyTorch's predefined ImageNet DataSet class.

`contrastive_model_predict.py` is linear classifier training script. The linear classifier uses the SimCLR model's backbone encoder to train a classifier over the CUB-2011-200 dataset. This can be run using `python3 -t $train -m $model_file -i $prediction_image -l $classifier_save_file` where
- `-t` boolean, whether the model is being trained or the image is being evaluated
- `-m` filename, the model whose backbone encoder is being used
- `-i` filename, image being predicted
- `-l` filename, save location of trained classifier model weights

Several miscellaneous Python scripts are included as helpers. Their code is incorporated into the notebook or they are utility and their inclusion in the notebook is unecessary.