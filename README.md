# be-modeling
Code used for training BE-Hive models

Paper: https://doi.org/10.1016/j.cell.2020.05.037

This code is not intended as a turn-key package for training models, but instead represents an archival reference of code used to train the models. 

If you are looking to use pre-trained BE-Hive models, please refer to our interactive web app at https://www.crisprbehive.design or our pre-trained model repositories at https://github.com/maxwshen/be_predict_efficiency and https://github.com/maxwshen/be_predict_bystander.

For more details on model training, please refer to the Methods section of our paper.

## Versions used to run the code
* Python 3.7 and standard packages (pickle, scipy, numpy, pandas)
* scikit-learn==0.20.3
* Biopython==1.73
* pytorch==1.1.0 
* torchvision==0.2.2

# Modeling base editing efficiency
BE-Hive uses gradient-boosted regression trees for predicting base editing efficiency. The input to the model is a 50-nt sequence context where the 20-nt Cas9 sgRNA is positioned at index 21 through 40, such that there are 20 nucleotides 5' of the sgRNA and 10 nucleotides 3' of the sgRNA. The output is the fraction of sequenced reads that contain any substitution mutation from base editing, broadly between protospacer positions -10 and 20, among all sequenced reads. The output value is logit-transformed and normalized to a mean of 0 and standard deviation of 1.

For CBEs, we defined base editing activity as C to A, G, or T at protospacer positions 9 to 20 and G to A or C at positions 9 to 5. For ABEs, base editing activity was defined as A to G at positions 5 to 20, A to C or T at positions 1 to 10, and C to G or T at positions 1 to 10. 

The script `anyedit_gbtr.py` is used to train the efficiency models.

The processed data in the data format used to train BE-Hive's efficiency model is available at https://figshare.com/articles/dataset/Processed_editing_efficiency_data/10673816.

For details on transforming the logit mean 0 output values back to the fraction of sequenced reads that contain any substitution mutation from base editing among all sequenced reads, please refer to the documentation in our interactive web app at https://www.crisprbehive.design/guide.


# Modeling base editing bystander patterns
BE-Hive uses a deep conditional autoregressive model to predict base editing bystander patterns. The input to the model is a 50-nt sequence context where the 20-nt Cas9 sgRNA is positioned at index 21 through 40, such that there are 20 nucleotides 5' of the sgRNA and 10 nucleotides 3' of the sgRNA. The output is a frequency distribution over exponentially many sequences that contain at least one base editing substitution mutation somewhere in the target sequence, typically from protospacer positions -5 to 20.

For CBEs, we defined base editing activity as C to A, G, or T at protospacer positions 9 to 20 and G to A or C at positions 9 to 5. For ABEs, base editing activity was defined as A to G at positions 5 to 20, A to C or T at positions 1 to 10, and C to G or T at positions 1 to 10. 

The script `c_ag_res.py` is used to train the bystander models for cytosine base editors (CBEs), and the script `c_ag_res_abe.py` is used for adenine base editors (ABEs).

The processed data in the data format used to train BE-Hive's efficiency model is available at https://figshare.com/articles/dataset/Processed_bystander_editing_data/10678097.


# Notes on using this code to train models on new data
The core logic of the scripts can be surmised from their main() functions. Each script first loads in data, then trains a model, and saves the model and training statistics to an output file. The formatted input data used to train our published BE-Hive models are available at https://figshare.com/articles/dataset/Processed_editing_efficiency_data/10673816 and https://figshare.com/articles/dataset/Processed_bystander_editing_data/10678097. 

If you plan to use our model training code on new data, you will need to process your data into the appropriate data format assumed by our code. These formats are described above, and are .csv or .pkl file types. You can refer to the processed data in the figshare links, and read the parts of the code that read in the input data and process it, to better understand the data format expected by the modeling scripts.

The modeling scripts require sklearn and pytorch.