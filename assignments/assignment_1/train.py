import time
import argparse
import torch 
import torchvision.transforms.v2 as v2
from torch import optim
import torchvision.models as models
from dlvc.models.vit import VisionTransformerShallow as VTS, VisionTransformerDeep as VTD, VisionTransformerDeepResidual as VTDR
from dlvc.models.cnn import SimpleCNN as SCNN, DeepCNN as DCNN, DeepNormalizedCNN as DNCNN
from dlvc.evaluation import cifar_load, train_model_opt, save_metrics, plot_metrics, comparison

# filter warnings
import warnings
warnings.filterwarnings('ignore')

dropout_rate = 0.5

weight_decay = 0.0001

lr_rate = 0.001

        
transform_train = v2.Compose([
    v2.ToImage(), 
    v2.RandomHorizontalFlip(),
    v2.RandomCrop(32, padding=4),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])
])

transform_val = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])
])

train_data_opt, val_data_opt, test_data_opt = cifar_load(transform_train, transform_val)


def main(model, name):

    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    if model == SCNN() or model == DCNN() or model == DNCNN():
    
        print(f'\nTraining model with hyperparameters: dropout_rate={dropout_rate}, weight_decay={weight_decay} and learning_rate={lr_rate}')
    else:
        print(f'\nTraining model with hyperparameters: dropout_rate={dropout_rate}, weight_decay={weight_decay} and learning_rate={lr_rate}')

    trainer = train_model_opt(model, optimizer, name, train_data_opt, val_data_opt)

    start_time = time.time()

    trainer.train()

    end_time = time.time()

    training_time = end_time - start_time

    print(f"\nTraining time: {training_time} seconds")

    save_metrics(trainer, name, training_time)

    plot_metrics(trainer, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model on CIFAR-10')
    # model and name argument
    parser.add_argument('--model', type=str, required=True, default='SimpleCNN', help='Model to train (SCNN, DCNN, DNCNN, VTS, VTD, VTDR, RN18)')
    parser.add_argument('--name', type=str, required=True, default='SimpleCNN', help='Name of the model to save')

    args = parser.parse_args()

    if args.model == 'SCNN':
        model = SCNN(dropout_rate)
    elif args.model == 'DCNN':
        model = DCNN(dropout_rate)
    elif args.model == 'DNCNN':
        model = DNCNN(dropout_rate)
    elif args.model == 'VTS':
        model = VTS(dropout_rate)
    elif args.model == 'VTD':
        model = VTD(dropout_rate)
    elif args.model == 'VTDR':
        model = VTDR(dropout_rate)
    elif args.model == 'RN18':
        model = models.resnet18()
    else:
        raise ValueError(f'Unknown model: {args.model}')
    

    main(model, args.name)


# Run the script
# python train.py --model SCNN --name SCNN