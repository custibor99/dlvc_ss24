# run all prompt commands
# Usage: ./execute.sh

# Train all models

models_cnn=("SCNN" "DCNN" "DNCNN" "SCNN_opt" "DCNN_opt" "DNCNN_opt")
models_vit=("VTS" "VTD" "VTDR" "VTS_opt" "VTD_opt" "VTDR_opt")
models_res=("RN18" "RN18_opt")

# Merge the arrays
all_models=("${models_cnn[@]}" "${models_vit[@]}" "${models_res[@]}")

total=${#all_models[@]}

counter=1

for model in "${models_cnn[@]}"
do  
    echo ""
    echo "Training CNN model $counter/$total: $model"
    python train_CNN.py --model $model --name $model
    ((counter++))
done

for model in "${models_vit[@]}"
do
    echo ""
    echo "Training ViT model $counter/$total: $model"
    python train_Vit.py --model $model --name $model
    ((counter++))
done

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
done