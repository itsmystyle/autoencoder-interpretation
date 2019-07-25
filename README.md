# How to train
1. Download PubMed 20k/200k RCT numbers replaced with at sign from https://github.com/Franck-Dernoncourt/pubmed-rct .
2. Put glove.6B.300d.txt, train.txt, dev.txt and test.txt into the data/ .
3. cd src/ and run the following command to preprocess and generate training data.
```
python make_dataset.py ../data/
```

4. Run the following command to prepapre model folder. Feel free to open the config.json to tune some hyperparameters.
```
mkdir ../models/your_model_folder
cp ../models/hnn/config.json ../models/your_model_folder
```

5. Run the following command to start training.
```
python train.py ../models/your_model_folder/
```

6. Trained model will saved in the folder 'your_model_folder/'



# How to predict
## Download pretrained model
1. Run the following shell
```
./download.sh
```
2. cd src/

## To predict test.txt
1. Run the following command to predict test.txt. You are able to change x in '--epoch x' to the best epochs you ran.
```
python predict.py ../models/hnn/ --epoch 7 --input_mode 1 --input_dir ../data/test.pkl --output_dir result.txt

or

python predict.py ../models/your_model_folder/ --epoch 7 --input_mode 1 --input_dir ../data/test.pkl --output_dir result.txt
```

## To predict abstract
1. Run the following command to predict input.
```
python predict.py ../models/hnn/ --epoch 7

or

python predict.py ../models/your_model_folder/ --epoch 7
```

2. Type in the abstract and it will output the result. Try input the following abstract.
```
To evaluate the performance ( efficacy , safety and acceptability ) of a new micro-adherent absorbent dressing ( UrgoClean ) compared with a hydrofiber dressing ( Aquacel ) in the local management of venous leg ulcers , in the debridement stage .$$$A non-inferiority European randomised controlled clinical trial ( RCT ) was conducted in @ centres , on patients presenting with venous or predominantly venous , mixed aetiology leg ulcers at their sloughy stage ( with more than @ % of the wound bed covered with slough at baseline ) .$$$Patients were followed over a @-week period and assessed weekly .$$$The primary judgement criteria was the relative regression of the wound surface area after the @-week treatment period .
```
