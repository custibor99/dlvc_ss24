import argparse
import torch 
import torchvision.transforms.v2 as v2
from torch import optim
from dlvc.models.cnn import SimpleCNN as SCNN, DeepCNN as DCNN, DeepNormalizedCNN as DNCNN
from dlvc.models.cnn_opt import SimpleCNN as SCNN_opt, DeepCNN as DCNN_opt, DeepNormalizedCNN as DNCNN_opt
from dlvc.evaluation import cifar_load, train_model_opt, save_metrics, plot_metrics

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
    
        print(f'\n\Training model with hyperparameters: dropout_rate={dropout_rate}, weight_decay={weight_decay} and learning_rate={lr_rate}')
    else:
        print(f'\n\Training model with hyperparameters: dropout_rate={dropout_rate}, weight_decay={weight_decay} and learning_rate={lr_rate}')

    trainer = train_model_opt(model, optimizer, name, train_data_opt, val_data_opt)

    trainer.train()

    save_metrics(trainer, name)

    plot_metrics(trainer, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model on CIFAR-10')
    # model and name argument
    parser.add_argument('--model', type=str, required=True, default='SimpleCNN', help='Model to train (SCNN, DCNN, DNCNN, SCNN_opt, DCNN_opt, DNCNN_opt)')
    parser.add_argument('--name', type=str, required=True, default='SimpleCNN', help='Name of the model to save')

    args = parser.parse_args()


    if args.model == 'SCNN':
        model = SCNN()
    elif args.model == 'DCNN':
        model = DCNN()
    elif args.model == 'DNCNN':
        model = DNCNN()
    elif args.model == 'SCNN_opt':
        model = SCNN_opt(dropout_rate)
    elif args.model == 'DCNN_opt':
        model = DCNN_opt(dropout_rate)
    elif args.model == 'DNCNN_opt':
        model = DNCNN_opt(dropout_rate)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    

    main(model, args.name)

# Run the script
# python train_CNN.py --model DCNN_opt --name DCNN_opt