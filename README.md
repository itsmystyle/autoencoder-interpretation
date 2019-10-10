
# How to train

1. Put english corpus (train.txt, dev.txt, test.txt and long_test.txt[optional]) into the **data/**
2. Edit **data/config.json**
3. Change directory to **src/** and run the following command to preprocess and generate training data.

    ```bash
    python make_dataset.py ../data/
    ```

4. Run the following command to prepapre model folder. Feel free to open the **models/seq2seq/config.json** to tune some hyperparameters.

    ```bash
    mkdir ../models/<your_model_folder>
    cp ../models/seq2seq/config.json ../models/<your_model_folder>
    ```

5. Run the following command to start training.

    ```bash
    python train.py ../models/<your_model_folder>/
    ```

6. Trained model will saved in the folder **models/<your_model_folder>/**
