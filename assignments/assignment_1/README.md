<p align="center">
  <img src="TU_Logo.png" alt="TU Logo"/>
</p>

## Deep Learning for Visual Computing

### Assignment 1

#### Setup

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

#### Models

CNN models:
- SCNN : Simple Convolutional Neural Network with dropout
- DCNN : Deep Convolutional Neural Network with dropout
- DNCNN : Deep Normalized Convolutional Neural Network with dropout

ViT models:
- VTS : Vision Transformer Small with dropout
- VTD : Vision Transformer Deep with dropout
- VTDR : Vision Transformer Deep with Residual connections and dropout

ResNet models:
- RN18 : Inherit ResNet18

#### Starting up

Move to the directory containing the `train.py` and `test.py` files.

```bash
cd group_40
```

#### Train a specific model

To train a specific model, run the following command:
```bash
python train.py --model <algorithm> --name <name_to_save>
```

- `algorithm` can be any of the models mentioned above.
- `name_to_save` is the name of the model to save. Please note that the model will be saved in the `dlvc/weights` directory with the name `<name_to_save>.pt`.

The training and evaluation results will be saved in the `results.csv` file with attributes such as `model`,`train/loss`,`train/mClassAcc`,`train/mAcc`,`val/loss`,`val/mClassAcc`,`val/mAcc` where `mClassAcc` is the mean class accuracy and `mAcc` is the mean accuracy.

The related training graphs can be found in the `img` directory.

To train all models, run the following command:
```bash
bash train_test_all.sh
```

For more specific experiments, you can modify the `train_test_all.sh` file or change the relevant parameters in the `train.py` file.

#### Test a specific model

To test a specific model, run the following command:
```bash
python test.py --model <algorithm> --name <name_to_save>
```

- `algorithm` can be any of the models mentioned above.
- `name_to_save` is the name of the model to save. Please note that the model will be loaded from the `dlvc/weights` directory with the name `<name_to_save>.pt`.

The test results will be saved in the `results.csv` file with attributes such as `test/loss`,`test/mClassAcc`,`test/mAcc` on the same row with training results for the `model`.

#### Results

The results of the experiments can be found in the `results.csv` file. To compare the results, you can use the `results.csv` file or the training graphs in the `img` directory.
A comparison of the models can be generated in tabular form using the following command:

```bash
python -c 'from dlvc.evaluation import comparison; comparison()'
```