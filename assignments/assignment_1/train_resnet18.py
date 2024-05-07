import argparse
import torch
from torch import optim
import torchvision.transforms.v2 as v2
import torchvision.models as models
from dlvc.models.resnet18_opt import ResNet18opt as RN18_opt
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

    if model == models.resnet18():

        print(f'\nTraining model with hyperparameters: weight_decay={weight_decay} and learning_rate={lr_rate}')

    else:
    
        print(f'\nTraining model with hyperparameters: dropout_rate={dropout_rate}, weight_decay={weight_decay} and learning_rate={lr_rate}')

    trainer = train_model_opt(model, optimizer, name, train_data_opt, val_data_opt)

    trainer.train()

    save_metrics(trainer, name)

    plot_metrics(trainer, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet18 model on CIFAR-10')
    parser.add_argument('--model', type=str, required=True, help='Model to train (ResNet18, ResNet18opt)')
    parser.add_argument('--name', type=str, required=True, help='Name of the model to save')

    args = parser.parse_args()

    if args.model == 'RN18':
        model = models.resnet18()
    elif args.model == 'RN18_opt':
        model = RN18_opt(dropout_rate)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    main(model, args.name)

# Run the script
# python train_ViT.py --model ResNet18opt --name ResNet18opt --device cpu
