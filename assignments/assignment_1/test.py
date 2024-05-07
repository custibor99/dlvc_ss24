import argparse
import torch 
import torchvision.transforms.v2 as v2
import torchvision.models as models
from dlvc.models.vit import VisionTransformerShallow as VTS, VisionTransformerDeep as VTD, VisionTransformerDeepResidual as VTDR
from dlvc.models.cnn import SimpleCNN as SCNN, DeepCNN as DCNN, DeepNormalizedCNN as DNCNN
from dlvc.evaluation import cifar_load, test_model

# filter warnings
import warnings
warnings.filterwarnings('ignore')
        
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

    model, test_metric = test_model(model, name, test_data_opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a model on CIFAR-10')
    parser.add_argument('--model', type=str, required=True, help='Model to train (VTS, VTD, VTDR, SCNN, DCNN, DNCNN, ResNet18)')
    parser.add_argument('--name', type=str, required=True, help='Name of the model to save')
    args = parser.parse_args()

    if args.model == 'VTS':
        model = VTS()
    elif args.model == 'VTD':
        model = VTD()
    elif args.model == 'VTDR':
        model = VTDR()
    elif args.model == 'SCNN':
        model = SCNN()
    elif args.model == 'DCNN':
        model = DCNN()
    elif args.model == 'DNCNN':
        model = DNCNN()
    elif args.model == 'RN18':
        model = models.resnet18()
    else:
        raise ValueError(f'Unknown model: {args.model}')
    

    main(model, args.name)

# Run the script
# python test.py --model RN18_opt --name RN18_opt