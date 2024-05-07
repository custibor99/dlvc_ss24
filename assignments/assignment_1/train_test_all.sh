# run all prompt commands
# Usage: ./train_test_all.sh

models_cnn=("SCNN" "DCNN" "DNCNN")
models_vit=("VTS" "VTD" "VTDR")
models_res=("RN18")

all_models=("${models_cnn[@]}" "${models_vit[@]}" "${models_res[@]}")

total=${#all_models[@]}

counter=1
# Train all models
for model in "${all_models[@]}"
do  
    echo ""
    echo "Training model $counter/$total: $model"
    python train.py --model $model --name $model
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

echo ""
echo "All models trained and tested successfully!"
echo ""

echo "Comparing test results and generating evaluation plots(img/evaluation.png)"
echo ""
python -c 'from dlvc.evaluation import comparison; comparison()'
echo ""
echo "Evaluation plots generated successfully!"
echo ""

echo "All tasks completed successfully!"
echo ""
echo "Exiting..."
echo ""
exit 0
