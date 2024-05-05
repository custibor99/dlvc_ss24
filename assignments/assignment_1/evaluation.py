from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from dlvc.datasets.cifar10 import  CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from dlvc.metrics import Accuracy
import torchvision.transforms.v2 as v2
from torch import optim
import torch 
from dlvc.trainer import ImgClassificationTrainer

from torch.utils.data import DataLoader
import torch.nn as nn

import ast

def cifar_load(transformer):
    train_data_opt = CIFAR10Dataset("data/cifar-10-python/cifar-10-batches-py", Subset.TRAINING, transformer)
    val_data_opt = CIFAR10Dataset("data/cifar-10-python/cifar-10-batches-py",Subset.VALIDATION, transformer)
    test_data_opt = CIFAR10Dataset("data/cifar-10-python/cifar-10-batches-py", Subset.TEST, transformer)
    return train_data_opt, val_data_opt, test_data_opt

def train_model_opt(model, optimizer, name, train_data_opt, val_data_opt):
    print("\nTraining model: ", model.__class__.__name__)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_metric = Accuracy(classes=train_data_opt.classes)
    val_metric = Accuracy(classes=val_data_opt.classes)
    device = torch.device("cpu")

    trainer = ImgClassificationTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data_opt,
        val_data_opt,
        device,
        num_epochs = 10,
        # for the model name, we use the class name of the model + the hyperparameters
        training_save_dir="dlvc/weights/opt/" + name+ ".pt",
        batch_size=64,
        val_frequency=1
    )
    return trainer

def test_model(model, params, test_data_opt):
    print("\nTesting model: ", model.__class__.__name__)

    model.eval()

    # Define device
    device = torch.device("cpu")

    # Initialize metric for test set
    test_metric = Accuracy(classes=test_data_opt.classes)

    # Create DataLoader for test set
    test_loader_opt = DataLoader(test_data_opt, batch_size=64, shuffle=False)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize loss and correct predictions
    test_loss = 0.0
    correct_preds = {classname: 0 for classname in test_data_opt.classes}
    total_preds = {classname: 0 for classname in test_data_opt.classes}

    # Evaluate model on test set
    with torch.no_grad():
        for images, labels in test_loader_opt:
            images = images.float().to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(output, 1)
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_preds[test_data_opt.classes[label]] += 1
                total_preds[test_data_opt.classes[label]] += 1
            test_metric.update(output, labels)

    # Calculate average loss and accuracy
    test_loss /= len(test_loader_opt.dataset)
    print("Test Loss: ", test_loss)
    print("Test Accuracy:", test_metric.accuracy())
    print("Per-class Accuracy:", test_metric.per_class_accuracy())
    for classname, correct_count in correct_preds.items():
        accuracy = 100 * float(correct_count) / total_preds[classname]
        print("Accuracy for class {:5s} is: {:.1f} ".format(classname, accuracy))

    # Load existing results or create a new DataFrame
    results = pd.read_csv("results_opt.csv")
 
    mask = (results["model"] == model.__class__.__name__) & (results["params"] == params)
    if results[mask].empty:
        # Add model to results with concat
        results = pd.concat([results, pd.DataFrame({
            "model": [model.__class__.__name__],
            "test/loss": [test_loss],
            "test/mClassAcc": [test_metric.per_class_accuracy()],
            "test/mAcc": [test_metric.accuracy()],
            "params": [params]
        })], ignore_index=True)

    else:
        results.loc[results["model"] == model.__class__.__name__ & results["params"] == params, "test/loss"] = test_loss
        results.loc[results["model"] == model.__class__.__name__ & results["params"] == params, "test/mClassAcc"] = test_metric.per_class_accuracy()
        results.loc[results["model"] == model.__class__.__class__.__name__ & results["params"] == params, "test/mAcc"] = test_metric.accuracy()
        results.loc[results["model"] == model.__class__.__name__ & results["params"] == params, "params"] = params

    # Save results
    results.to_csv("results_opt.csv", index=False)
    
    return model, test_metric

def save_metrics(trainer, model_name, params):
    # Save metrics
    metrics_train = trainer.metrics_train
    metrics_train = list(zip(*metrics_train))
    loss_train = metrics_train[0]
    acc_train = metrics_train[1]
    class_acc_train = metrics_train[2]


    metrics_val = trainer.metrics_val
    metrics_val = list(zip(*metrics_val))
    loss_val = metrics_val[0]
    acc_val = metrics_val[1]
    class_acc_val = metrics_val[2]

    # read existing metrics or create new DataFrame
    try:
        metrics = pd.read_csv("metrics_opt.csv")
    except FileNotFoundError or pd.errors.EmptyDataError:
        metrics = pd.DataFrame(columns=["model", "train/loss", "train/mClassAcc", "train/mAcc", "val/loss", "val/mClassAcc", "val/mAcc", "params"])

    
    mask = (metrics["model"] == model_name) & (metrics["params"] == params)
    if metrics[mask].empty:
         # Add model to metrics with concat
        metrics = pd.concat([metrics, pd.DataFrame({
            "model": [model_name],
            "train/loss": [loss_train],
            "train/mClassAcc": [class_acc_train],
            "train/mAcc": [acc_train],
            "val/loss": [loss_val],
            "val/mClassAcc": [class_acc_val],
            "val/mAcc": [acc_val],
            "params": [params]
        })], ignore_index=True)
    else:
        metrics.loc[metrics["model"] == model_name & metrics["params"] == params, "train/loss"] = loss_train
        metrics.loc[metrics["model"] == model_name & metrics["params"] == params, "train/mClassAcc"] = class_acc_train
        metrics.loc[metrics["model"] == model_name & metrics["params"] == params, "train/mAcc"] = acc_train
        metrics.loc[metrics["model"] == model_name & metrics["params"] == params, "val/loss"] = loss_val
        metrics.loc[metrics["model"] == model_name & metrics["params"] == params, "val/mClassAcc"] = class_acc_val
        metrics.loc[metrics["model"] == model_name & metrics["params"] == params, "val/mAcc"] = acc_val
        metrics.loc[metrics["model"] == model_name & metrics["params"] == params, "params"] = params

    # Save metrics
    metrics.to_csv("metrics_opt.csv", index=False)

def plot_metrics(trainer, name):
    print("\nPlotting metrics for model: ", trainer.model.__class__.__name__)
    plt.style.use('ggplot')

    metrics = trainer.metrics_train
    metrics = list(zip(*metrics))
    loss = metrics[0]
    acc = metrics[1]
    class_acc = metrics[2]


    metrics_val = trainer.metrics_val
    metrics_val = list(zip(*metrics_val))
    loss_val = metrics_val[0]
    acc_val = metrics_val[1]
    class_acc_val = metrics_val[2]

    fig, ax = plt.subplots(1,3, figsize = (19,4))
    ax[0].plot(loss, label = "train")
    ax[0].plot(loss_val, label = "eval")
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')

    ax[1].plot(acc)
    ax[1].plot(acc_val)
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')

    ax[2].plot(class_acc)
    ax[2].plot(class_acc_val)
    ax[2].set_ylabel('Class Accuracy')
    ax[0].set_xlabel('Epoch')

    fig.legend()
    fig.suptitle("Metrics for model: " + name)
    fig.savefig("img/opt/"+name + ".png")

def plot_evaluation(df, title):
    plt.style.use('ggplot')
    # Create a color dictionary to map models to colors
    colors = {
            'DeepCNNOpt': 'red',
            'DeepNormalizedCNNOpt': 'blue', 
            'SimpleCNNOpt': 'green',
            'VisionTransformerShallowOpt': 'purple',
            'VisionTransformerDeepOpt': 'orange',
            'VisionTransformerDeepResidualOpt': 'brown',
            'ResNet18Dropout': 'pink'
        }

    fig, ax = plt.subplots(1, 3, figsize=(19, 4))
    handles = []
    labels = []

    for i, model in enumerate(df['model'].unique()):
        subset = df[df['model'] == model]

        train_loss = subset['train/loss'].apply(lambda x: [float(i) for i in x.strip('()[]').split(',')])
        val_loss = subset['val/loss'].apply(lambda x: [float(i) for i in x.strip('()[]').split(',')])

        # log scale loss
        train_loss = train_loss.apply(lambda x: [np.log(i) for i in x])
        val_loss = val_loss.apply(lambda x: [np.log(i) for i in x])

        train_acc = subset['train/mAcc'].apply(lambda x: [float(i) for i in x.strip('()[]').split(',')])
        val_acc = subset['val/mAcc'].apply(lambda x: [float(i) for i in x.strip('()[]').split(',')])

        train_class_acc = subset['train/mClassAcc'].apply(lambda x: [float(i) for i in x.strip('()[]').split(',')])
        val_class_acc = subset['val/mClassAcc'].apply(lambda x: [float(i) for i in x.strip('()[]').split(',')])

        # Plotting for loss, log scale
        ax[0].plot(train_loss.values[0], label=f"{model} - Train", color=colors[model], linestyle='-')
        ax[0].plot(val_loss.values[0], label=f"{model} - Validation", color=colors[model], linestyle='--')
        ax[0].set_ylabel('Loss(log)')

        # Plotting for accuracy
        ax[1].plot(train_acc.values[0], label=f"{model} - Train", color=colors[model], linestyle='-')
        ax[1].plot(val_acc.values[0], label=f"{model} - Validation", color=colors[model], linestyle='--')
        ax[1].set_ylabel('Accuracy')

        # Plotting for class accuracy
        ax[2].plot(train_class_acc.values[0], label=f"{model} - Train", color=colors[model], linestyle='-')
        ax[2].plot(val_class_acc.values[0], label=f"{model} - Validation", color=colors[model], linestyle='--')
        ax[2].set_ylabel('Class Accuracy')

        handles_, labels_ = ax[2].get_legend_handles_labels()
        
        #if not already in the list
        for handle, label in zip(handles_, labels_):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    plt.tight_layout()

    # Add title
    fig.suptitle(f'{title} Metrics for Best Models(max val/mAcc)', fontsize=16)

    # move the title to the top and add some space
    plt.subplots_adjust(
        top=0.85, 
        hspace=0.4,
        wspace=0.2
        )  # Adjust the wspace value as needed

    # Add x-y labels
    for i, axis in enumerate(ax):
        axis.set_xlabel('Epoch')

    # legend outside the plot right and vertical
    fig.legend(
        handles, 
        labels, 
        loc='center right', 
        bbox_to_anchor=(1.2, 0.5),  # Adjust the position as needed
        fancybox=True, 
        shadow=True, 
        ncol=1
    )
    plt.savefig(f'img/{title}.png', bbox_inches='tight')  # Save the figure with tight bounding box


def get_best_model(df, opt):
    # Parse the 'params' column
    df['params'] = df['params'].apply(ast.literal_eval)
    df['left_right_mirror'] = df['params'].apply(lambda x: x['left_right_mirror'])
    df['random_crop'] = df['params'].apply(lambda x: x['random_crop'])
    # Define the conditions
    conditions = [
        (df['left_right_mirror'] == True) & (df['random_crop'] == True),
        (df['left_right_mirror'] == False) & (df['random_crop'] == False)
    ]

    # Define the labels
    labels = ['mirrior_and_crop', 'no_augmentation']

    # Create a new column 'condition' in the DataFrame
    df['condition'] = np.select(conditions, labels, default='other')

    # for conditions not other
    df = df[df['condition'] != 'other']

    best_model_indices = df.groupby(['condition', 'model'])[opt].idxmax()

    # Select the corresponding rows from the df DataFrame
    best_models = df.loc[best_model_indices]

    # Reset index to ensure the DataFrame has a clean index
    best_models.reset_index(drop=True, inplace=True)

    # get mirror, crop and mirror_and_crop
    no_augmentation = best_models[best_models['condition'] == 'no_augmentation']
    mirror_and_crop = best_models[best_models['condition'] == 'mirrior_and_crop']

    return no_augmentation, mirror_and_crop

def plot_test_results_subplots(df):
    mirror_and_crop, no_augmentation = get_best_model(df, 'test/mAcc')
    fig, axs = plt.subplots(2, 1, figsize=(20, 13))

    dfs = [mirror_and_crop, no_augmentation]
    titles = ['No Augmentation', 'Left-Right Mirror and Random Crop Augmentation']

    # Define the y-axis limit
    y_limit = 3  

    for i, (df, title) in enumerate(zip(dfs, titles)):
        ax = axs[i]
        metrics = ['test/loss', 'test/mClassAcc', 'test/mAcc']
        metric_labels = ['Loss', 'Class Accuracy', 'Accuracy']
        metrics_colors = ['#FF6666', '#6699FF', '#33CC33']

        bar_width = 0.2
        model_indices = np.arange(len(df['model']))

        for j, metric in enumerate(metrics):
            ax.bar(model_indices + j * bar_width, df[metric], bar_width, label=metric_labels[j], color=metrics_colors[j])
            # Display bar values with 4 decimals
            for k, value in enumerate(df[metric]):
                ax.text(model_indices[k] + j * bar_width, min(value + 0.01, y_limit), '{:.3f}'.format(value), ha='center', va='bottom')

        ax.set_xticks(model_indices + bar_width)
        ax.set_xticklabels(df['model'])
        ax.set_ylabel('Metrics')
        ax.set_title(f'Test Results for {title}', y=1.05)
        ax.legend(loc='upper left')

        # Set y-axis limit
        ax.set_ylim(0, y_limit)

    # label font size
    plt.rcParams.update({'font.size': 14})

    # Adjust the top space
    plt.subplots_adjust(
        top=0.90, 
        hspace=0.4,
        wspace=0.2
        )  # Adjust the wspace value as needed

    # figure title
    fig.suptitle('Test Metrics for Best Models (max test/mAcc)', fontsize=16)

    plt.savefig(f'img/test_results_subplots.png', bbox_inches='tight')
