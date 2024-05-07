import ast
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from torch.utils.data import DataLoader
from dlvc.datasets.cifar10 import  CIFAR10Dataset
from dlvc.datasets.dataset import Subset

seed = 42
torch.manual_seed(seed)
plt.style.use('ggplot')

def cifar_load(transformer_train, transformer_val):
    train_data_opt = CIFAR10Dataset("data/cifar-10-python/cifar-10-batches-py", Subset.TRAINING, transformer_train)
    val_data_opt = CIFAR10Dataset("data/cifar-10-python/cifar-10-batches-py",Subset.VALIDATION, transformer_val)
    test_data_opt = CIFAR10Dataset("data/cifar-10-python/cifar-10-batches-py", Subset.TEST, transformer_val)
    return train_data_opt, val_data_opt, test_data_opt

def train_model_opt(model, optimizer, name, train_data_opt, val_data_opt):
    print("\nTraining model: ", name)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_metric = Accuracy(classes=train_data_opt.classes)
    val_metric = Accuracy(classes=val_data_opt.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        training_save_dir="dlvc/weights/" + name+ ".pt",
        batch_size=64,
        val_frequency=1
    )
    return trainer

def test_model(model, name, test_data_opt):
    print("\nTesting model: ", name, "\n")

    # load the model
    try:
        model.load_state_dict(torch.load("dlvc/weights/" + name + ".pt"))
        print(f'Model {name} loaded from weights folder\n')
    except:
        print(f'Model {name} not found in weights folder')
        return

    # evaluate the model
    model.eval()

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    start_time = time.time()
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
    end_time = time.time()

    testing_time = end_time - start_time

    print(f"Testing time: {testing_time} seconds\n")

    # Calculate average loss and accuracy
    test_loss /= len(test_loader_opt.dataset)
    print("Test Loss: ", test_loss)
    print("Test Accuracy:", test_metric.accuracy())
    print("Per-class Accuracy:", test_metric.per_class_accuracy())
    for classname, correct_count in correct_preds.items():
        accuracy = 100 * float(correct_count) / total_preds[classname]
        print("Accuracy for class {:5s} is: {:.1f} ".format(classname, accuracy))

    # exit() # uncomment this line when you don't want to change the results.csv file

    # Load existing results or create a new DataFrame
    results = pd.read_csv("results.csv")
    results['model'] = results['model'].astype(str)
    # Add test results to DataFrame where model name matches
    results.loc[results['model'] == name, 'test/loss'] = test_loss
    results.loc[results['model'] == name, 'test/mClassAcc'] = test_metric.per_class_accuracy()
    results.loc[results['model'] == name, 'test/mAcc'] = test_metric.accuracy()
    results.loc[results['model'] == name, 'test/time'] = testing_time

    # Save results
    results.to_csv("results.csv", index=False)
    
    return model, test_metric

def save_metrics(trainer, model_name, training_time):
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
        results = pd.read_csv("results.csv")
        results['model'] = results['model'].astype(str)
    except FileNotFoundError or pd.errors.EmptyDataError:
        results = pd.DataFrame(columns=["model", "train/loss", "train/mClassAcc", "train/mAcc", "val/loss", "val/mClassAcc", "val/mAcc", "test/loss", "test/mClassAcc", "test/mAcc", "test/time", "train/time"])

    # if model already exists in results, update the values
    if model_name in results['model'].values:
        results.loc[results['model'] == model_name, 'train/loss'] = str(loss_train)
        results.loc[results['model'] == model_name, 'train/mClassAcc'] = str(class_acc_train)
        results.loc[results['model'] == model_name, 'train/mAcc'] = str(acc_train)
        results.loc[results['model'] == model_name, 'val/loss'] = str(loss_val)
        results.loc[results['model'] == model_name, 'val/mClassAcc'] = str(class_acc_val)
        results.loc[results['model'] == model_name, 'val/mAcc'] = str(acc_val)
        results.loc[results['model'] == model_name, 'train/time'] = training_time
    else:
        # Add model to results with concat
        results = pd.concat([results, pd.DataFrame({
            "model": [model_name],
            "train/loss": [loss_train],
            "train/mClassAcc": [class_acc_train],
            "train/mAcc": [acc_train],
            "val/loss": [loss_val],
            "val/mClassAcc": [class_acc_val],
            "val/mAcc": [acc_val],
            "test/loss": [0],
            "test/mClassAcc": [0],
            "test/mAcc": [0],
            "train/time": [training_time],
            "test/time": [0]
        })], ignore_index=True)


    # Save results
    results.to_csv("results.csv", index=False)

    print("\nMetrics saved for model ", model_name, " in results.csv")

def plot_metrics(trainer, name):
    print("\nPlotting metrics for model: ", name, "\n")
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
    fig.savefig("img/"+name + ".png")

def plot_evaluation(df):
    plt.style.use('ggplot')
    # Create a color dictionary to map models to colors
    colors = {
        'SCNN': 'green',
        'DCNN': 'red',
        'DNCNN': 'blue',
        'VTS': 'purple',
        'VTD': 'orange',
        'VTDR': 'brown',
        'RN18': 'pink',
        }

    fig, ax = plt.subplots(1, 3, figsize=(19, 4))
    handles = []
    labels = []

    df['model'] = df['model'].astype(str)

    for i, model in enumerate(df['model']):
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
    fig.suptitle(f'Metrics for Models', fontsize=16)

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
        bbox_to_anchor=(1.13, 0.5),  # Adjust the position as needed
        fancybox=True, 
        shadow=True, 
        ncol=1
    )

    plt.savefig(f'img/evaluation.png', bbox_inches='tight')  # Save the figure with tight bounding box
    

def comparison():
    # Load results
    results = pd.read_csv('results.csv')
    plot_evaluation(results)
    # get mean of val/mAcc, val/mClassAcc
    results['val/mAcc'] = results['val/mAcc'].apply(lambda x: ast.literal_eval(x))
    results['val/mClassAcc'] = results['val/mClassAcc'].apply(lambda x: ast.literal_eval(x))
    results['val/mAcc'] = results['val/mAcc'].apply(lambda x: sum(x)/len(x))
    results['val/mClassAcc'] = results['val/mClassAcc'].apply(lambda x: sum(x)/len(x))

    # Prepare data for tabulate
    table_data = []
    for index, row in results.iterrows():
        table_data.append([row['model'], f"{row['val/mAcc']:.4f}", f"{row['val/mClassAcc']:.4f}", f"{row['test/mAcc']:.4f}", f"{row['test/mClassAcc']:.4f}", f"{row['train/time']:.2f}", f"{row['test/time']:.2f}"])

    # Print table with tabulate
    print(tabulate(table_data, headers=["Model", "val ACC", "val PACC", "test ACC", "test PACC", "train time(s)", "test time(s)"], tablefmt="pipe"))

# run the script for comparison
# python -c 'from evaluation import comparison; comparison()'