# [Source-Free Cross-Domain State of Charge Estimation of Lithium-ion Batteries at Different Ambient Temperatures](https://ieeexplore.ieee.org/document/10058040)
# Usage
* conda environment   
```
conda env create -f env.yaml
```
* Dataset  

more dataset for LIBs can be downloaded from [HERE](https://docs.google.com/spreadsheets/d/10w5yXdQtlQjTTS3BxPP233CiiBScIXecUp2OQuvJ_JI/edit#gid=0)
* Data processing  

put your data fold in ```normalized_data/``` and run this code  
```
python normalized_data/dataprocess.py
```
* To pretrain source model (including two different estimators)    
```
python run.py --mode pretrain --mkdir [your_folder] --source_data_path [] --source_temp [] --epochs --batch_size
```
(check run.py for more arguments)  
The model is saved in ```run/your_folder/saved_model/best.pt```
* Pseudo label    

Use pre-trained source model to generate pseudo labels for target data:    
```
python pseudo.py --temp --model --file
```
* To transfer a model  
```
python run.py --mode train --mkdir [] --source_data_path --source_temp --target_data_path --target_temp --epochs --batch_size
```
(check run.py for more arguments)   
* models  

We have provided pretrained models and models retrained only with limitted target labels for five temperatures of Panasonic 18650PF dataset in folder "models" for comparison.
* To test a model  
```
python run.py --mode test --mkdir [] --test_set [] --target_temp []
```
