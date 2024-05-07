<!-- # run all prompt commands
# Usage: ./execute.sh

# Train all models

models_cnn=("SCNN" "DCNN" "DNCNN" "SCNN_opt" "DCNN_opt" "DNCNN_opt")
models_vit=("VTS" "VTD" "VTDR" "VTS_opt" "VTD_opt" "VTDR_opt")
models_res=("RN18" "RN18_opt")

# Merge the arrays
all_models=("${models_cnn[@]}" "${models_vit[@]}" "${models_res[@]}")

total=${#all_models[@]}

# for model in "${models_cnn[@]}"
# do  
#     echo ""
#     echo "Training CNN model $counter/$total: $model"
#     python train_CNN.py --model $model --name $model
#     ((counter++))
# done

# counter=1

# for model in "${models_vit[@]}"
# do
#     echo ""
#     echo "Training ViT model $counter/$total: $model"
#     python train_Vit.py --model $model --name $model
#     ((counter++))
# done

for model in "${models_res[@]}"
do  
    echo ""
    echo "Training ResNet18 model $counter/$total: $model"
    python train_resnet18.py --model $model --name $model
    ((counter++))
done

counter=1
# Test all models
for model in "${all_models[@]}"
do
    echo ""
    echo -e "Testing model $counter/$total: $model"
    python test.py --model $model --name $model
    ((counter++))
done -->

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
- SCNN : Simple Convolutional Neural Network
- DCNN : Deep Convolutional Neural Network
- DNCNN : Deep Normalized Convolutional Neural Network
- SCNN_opt : Simple Convolutional Neural Network with dropout
- DCNN_opt : Deep Convolutional Neural Network with dropout
- DNCNN_opt : Deep Normalized Convolutional Neural Network with dropout

ViT models:
- VTS : Vision Transformer Small
- VTD : Vision Transformer Deep
- VTDR : Vision Transformer Deep with Residual connections
- VTS_opt : Vision Transformer Small with dropout
- VTD_opt : Vision Transformer Deep with dropout
- VTDR_opt : Vision Transformer Deep with Residual connections and dropout

ResNet models:
- RN18 : ResNet18
- RN18_opt : ResNet18 with dropout

#### Starting up

Move to the directory containing the `train_CNN.py`, `train_Vit.py`, `train_resnet18.py`, and `test.py` files.

#### Train a specific model

To train a specific model, run the following command:
```bash
python train_<model_type>.py --model <algorithm> --name <name_to_save>
```

- `model_type` can be `CNN`, `Vit`, or `resnet18`.
- `algorithm` can be any of the models mentioned above.
- `name_to_save` is the name of the model to save. Please note that the model will be saved in the `dlvc/weights` directory with the name `<name_to_save>.pt`.

The training and evaluation results will be saved in the `dlvc/results` directory with attributes such as `model`,`train/loss`,`train/mClassAcc`,`train/mAcc`,`val/loss`,`val/mClassAcc`,`val/mAcc` where `mClassAcc` is the mean class accuracy and `mAcc` is the mean accuracy.

#### Test a specific model

To test a specific model, run the following command:
```bash
python test.py --model <algorithm> --name <name_to_save>
```

- `algorithm` can be any of the models mentioned above.
- `name_to_save` is the name of the model to save. Please note that the model will be loaded from the `dlvc/weights` directory with the name `<name_to_save>.pt`.

The test results will be saved in the `dlvc/results` directory with attributes such as `test/loss`,`test/mClassAcc`,`test/mAcc` on the same row with training results for the `model`.