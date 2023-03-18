##  Crimean Plants Classification

![Example][example]<br>

## About the project
The project solves the task of image classification into 50 classes of plants.

In the process of finding the optimal model for **transfer learning**, 
the **MobileNetV3_Large** model was chosen, which was pre-trained 
on the ImageNet dataset. Currently, it has an accuracy 
of 0.99 on the test sample of 1500 plant images.

The **dataset** for training the model was **collected automatically**
 by parsing Yandex Images and includes a total of 13,723 plant images. More information about the dataset can be found [in this section](#dataset).


<details>
    <summary><strong>Project structure</strong></summary>

This is the structure of the project **after** running the `download_files.py` script, 
which downloads the checkpoint and the dataset:

```
.
├── checkpoints                                 
│   └── plants_model_checkpoint
│       └── plants_model.pth.tar                # Model checkpoint
│
├── data                                        
│   ├── dataset                                 
│   │   ├── test                                # Test image directory
│   │   │   ├── Adenophora liliifolia
│   │   │   ├── Adiantum capillus-veneris
│   │   │               ...
│   │   ├── train                               # Training image directory
│   │   │   ├── Adenophora liliifolia
│   │   │   ├── Adiantum capillus-veneris
│   │   │               ...
│   │   ├── test_labels.csv                     # Test CSV file
│   │   └── train_labels.csv                    # Training CSV file
│   └── crimean_plants.csv                      # CSV file containing plant names and class label
│
├── dataset_preparation                         
│   ├── parsing
│   │   ├── data
│   │   │   └── crimean_plants.csv              # CSV file containing plant names and class label
│   │   ├── geckodriver
│   │   │   └── geckodriver.exe                 # Driver for controlling FireFox
│   │   ├── parsing.py                          
│   │   ├── parsing_utils.py                    # Utilities required for parsing
│   │   └── yandex_images_parser.py             # Parser. (My parser project - gitlab.com/glebtutik/yandex_images_parser)
│   ├── cleaning                                 
│   │   ├── cleaning.py                         # Script for cleaning the dataset
│   │   └── cleaning_utils.py                   # Utilities required for cleaning the dataset
│   └── labeling
│       ├── labeling.py                         # Data labeling script
│       └── labeling_utils.py                   # Utilities required for cleaning the data
│
├── config.py                                   # Configurations
├── dataset.py                                  # Dataset class
├── download_files.py                           # Script for downloading the dataset and model checkpoint
├── hyperparameter_optimization.py              # Script for searching for optimal hyperparameters
├── model.py                                    # Model class
├── train.py                                    # Model training script
├── utils.py                                    # Utilities
├── requirements.txt                            # Project requirements
└── README.md
```
</details>


## Technical solutions

#### Main technologies

- **[Python 3.9](https://www.python.org/downloads/release/python-390/)**<br> 
- **[PyTorch](https://pytorch.org)** - deep learning framework.<br>
- **[TensorBoard](https://pytorch.org/docs/stable/tensorboard.html)** - real-time tracking of the training process.<br> 
- **[Optuna](https://optuna.org)** - search for optimal hyperparameters.<br>
- **[Selenium](https://www.selenium.dev)** - dataset parsing from Yandex Images.
- **[Pandas](https://pandas.pydata.org)** - work with CSV files.<br><br>


#### Transfer learning

Since the task was to achieve high accuracy on a small dataset,
the decision was made
to use a **pretrained model**.

<details>
    <summary><strong>List of models tested</strong></summary>

```python
models = ["alexnet", "convnext_tiny", "densenet121", "densenet201", "resnet18", "mobilenet_v3_small", "mobilenet_v3_large"]
```

</details>

Out of the selected 7 models, the most accurate results were shown by [ConvNeXt_Tiny](https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html),
 which is available by default in torchvision. However, due to its weight, preference was 
given to the [MobileNet_V3_Large](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html) 
model, which demonstrated comparable accuracy results, but was **5.2 times lighter in terms of parameters**.

The last fully connected layer of the classifier in MobileNet was replaced: 
```python
# Replace the last layer:
in_features = mobilenet.classifier[-1].in_features
mobilenet.classifier[-1] = nn.Linear(in_features=in_features, out_features=config.OUT_FEATURES)
```

Since the network was initially trained on the ImageNet dataset, which contains
1000 classes, its number of outputs was equal to a thousand.
After the layer was replaced, their number became equal to 50.
The ability to "freeze" gradients in the feature extractor was added.<br>
Otherwise, the model was left unchanged.<br><br>


#### Image Parsing and Dataset Preparation

The [Yandex Images Parser](https://github.com/glebtk/yandex_images_parser) 
I previously wrote was used for parsing images.<br>

For a particular plant, a search was made for 200 images based on its Latin name. 
Then, for each of the first 20 search results, a search was made for 40 similar images.

At the **cleaning** stage of the dataset, all images went through a check
 for compliance with the conditions, duplicates were removed.
After that, the images were resized.
	
More detailed about **data cleaning algorithm** you can read here:

<details>
    <summary><strong>Data Cleansing Algorythm</strong></summary>

For images belonging to the same class, the following steps are applied:

1. Check if all images are valid, remove garbage.
2. Remove images that do not meet the specified conditions (size >= 256 pixels on the smaller side, RGB color mode)
3. Remove duplicate images. To do this, using the [ImageHash](https://pypi.org/project/ImageHash/) library, we obtain the [average_hash](https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html) of the images and delete images with the same or similar hashes.
4. Scale the images to 256px on the smaller side.
5. Save the images in a separate directory.
6. Done! Move on to the next class.

The final cleaning is performed manually.

</details>

The images were divided into training and test datasets during the **labeling** process. 
The paths to the images and class labels were saved in a CSV file.

## Dataset

The dataset contains **13,723 color images** of plants belonging to 50 
classes. Out of these, **12,223** are in the training set, and **1,500** in the test set.

Example of random images from the dataset:

![Dataset](https://i.imgur.com/1ZqRspx.png)

- Each image belongs to **only one** class.
- On average, there are **274** images per class.
- Minimum image size: **256x256 px**.
- Mean across the dataset: `[0.4074, 0.4307, 0.2870]`.
- Standard deviation: `[0.2128, 0.2006, 0.2053]`.


<details>
    <summary><strong>Dataset Structure</strong></summary>

```
dataset                                 
├── test                                        # Test images directory
│   ├── Adenophora liliifolia                   # Class 1 directory
│   │   ├──adenophora_liliifolia_test_1.jpg     
│   │   ├──adenophora_liliifolia_test_2.jpg
│   │               ...
│   │               
│   ├── Adiantum capillus-veneris               # Class 2 directory
│               ...
│
├── train                                       # Training images directory
│   ├── Adenophora liliifolia                   # Class 1 directory
│   │   ├──adenophora_liliifolia_train_1.jpg
│   │   ├──adenophora_liliifolia_train_2.jpg
│   │               ...
│   │
│   ├── Adiantum capillus-veneris               # Class 2 directory
│               ...
│
├── test_labels.csv                             # Test CSV file containing columns path and label
└── train_labels.csv                            # Training CSV file containing columns path and label
```

</details>

The dataset is available through the following direct link: 
[dataset.zip](https://gitlab.com/glebtutik/crimean_plants_classification_files/-/raw/main/data/dataset.zip)
(~274MB).

## Training Process and Results

The following hyperparameters were found during the search:

```python
BATCH_SIZE = 90
LEARNING_RATE = 9e-05
```

The model was trained for 20 epochs. The best test accuracy was **0.99**, and was
achieved after the 18th epoch.

Below is the confusion matrix and training model graphs.

<details>
    <summary><strong>Graphs and matrix</strong></summary>

**Graphs** obtained during training (accuracy, loss):

![Accuracy, Loss](https://i.imgur.com/caJE8S3.jpg)

**Confusion matrix**:

![Confusion matrix](https://i.imgur.com/XGxscVO.jpg)

</details>


## How to launch this project?

1. Clone the repository:
```bash
$ git clone https://github.com/glebtk/crimean_plants_classification.git
```

2. Before using it, you need to install the project requirements.<br>
Go to the cloned repository directory

```bash
$ cd path/to/directory
```

and run this command:

```bash
$ pip install -r requirements.txt
```

3. Make sure that all dependencies are successfully installed.
4. Execute the script `download_files.py `. It will load the dataset and
the checkpoint of the model in the right places:

```bash
$ python download_files.py
```

5. Done! You can run the project.

## Contact information 

[![Mail](https://i.imgur.com/HILZFT2.png)](mailto:tutikgv@gmail.com)
**E-mail:**
[tutikgv@gmail.com](mailto:tutikgv@gmail.com) <br>

[![Telegram](https://i.imgur.com/IMICyTA.png)](https://t.me/glebtutik)
**Telegram:**
https://t.me/glebtutik <br>



[cover]: https://i.imgur.com/D7So6VS.png "Cover"
[example]: https://i.imgur.com/jmO3u0U.gif "Example"


