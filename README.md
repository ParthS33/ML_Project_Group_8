Requirements:
* pytorch: 2.2.2
* python: 3.10
* tensorflow:
* numpy: 1.26.4
* nltk: 3.8.1
* skikit-learn: 1.4.2
* tqdm: 4.66.2
* matplotlib
* seaborn

The code should work with these. If it doesn't then we also have the requirements.txt which you can use to download the libraries.
GPU is required. You can download the CUDA toolkit from [here](https://developer.nvidia.com/cuda-toolkit). We used NVIDIA CUDA 12.1

# Step 1: Convert tensorflow to pytorch model
Download BERT-Base [(Google's pre-trained models)](https://github.com/google-research/bert)) in the project directory and unzip it.
Use the below code to convert a tensorflow checkpoint to a pytorch model.
Check out the README from the above link to find the accurate BERT model

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```
# Step 2: Training the model
Example script
```
python run_classifier_TABSA.py --task_name sentihood_NLI_M --data_dir data/sentihood/bert-pair/ --vocab_file uncased_L-12_H-768_A-12/vocab.txt --bert_config_file uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin --eval_test --do_lower_case --max_seq_length 128 --train_batch_size 24 --learning_rate 2e-5 --num_train_epochs 4.0 --output_dir results/sentihood/NLI_M --seed 42 --do_save_model
```
You can change the hyperparameters and tasks in the above script
If you want to use the scripts that we have used, then check out the Script.txt

# Step 3: Evaluation
Evaluation is performed on the test dataset

To evaulate all the models run the below code
```
python evaluation.py --pred_data_dir results/sentihood
```

To evaluate a particular model replace the path to the test epoch of the model
Example:
```
python evaluation.py --pred_data_dir results/sentihood/NLI_B_128_20_2e-05_4.0_1_1/test_ep_4.txt
```

# Step 4: Visualization
To be able to run the implemented methods, you first have to execute the script visualization.py

• To visualize the confusion matrix for a specific epoch, you can run the following command line (example for epoch 1).

plot_confusion_matrix(actual_dataset=test_NLI_M, prediction_dataset=test_ep_1, epoch_number=1)

• To plot the ROC curve with every epoch, and a specified class (‘None’ in the example), run the following code.

plot_roc_curve(actual_dataset = test_NLI_M, prediction_datasets = [test_ep_1, test_ep_2, test_ep_3, test_ep_4], class_name = 'None')

• To display the barplot of the F1 scores for each epoch, run the following code.

plot_f1_scores(actual_dataset=test_NLI_M, prediction_datasets=[test_ep_1, test_ep_2, test_ep_3, test_ep_4], epoch_names=['epoch 1', 'epoch 2', 'epoch 3', 'epoch 4'])
