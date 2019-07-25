# How to train
1. Put english corpus (train.txt, dev.txt, test.txt and long_test.txt[optional]) into the data/ .
2. Edit config.json .
3. cd src/ and run the following command to preprocess and generate training data .
```
python make_dataset.py ../data/
```

4. Run the following command to prepapre model folder. Feel free to open the config.json to tune some hyperparameters.
```
mkdir ../models/your_model_folder
cp ../models/seq2seq/config.json ../models/your_model_folder
```

5. Run the following command to start training.
```
python train.py ../models/your_model_folder/
```

6. Trained model will saved in the folder 'your_model_folder/'
