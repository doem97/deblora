"""
NAME: dota_linprob_v0.2.py
ENV: dift
VERSION: 0.2
AUTHOR: @doem1997

Description:
    This script performs linear probing on extracted features from the DOTA dataset.
    It trains a simple linear classifier or a ResNet18-based classifier on the
    extracted features and evaluates the model's performance on validation and test sets.

CHANGELOG:
    v0.2: - Optimize the logging and error handling
          - Optimize linear head: replace flatten with view
          - Add group-specific F1 score calculation (head, middle, tail)
          - Add support for abs path in train.csv file
    v0.1: - Implemented feature loading and dataset creation
          - Added linear classifier and ResNet18-based classifier options
          - Implemented training loop with validation
          - Added model evaluation and metric calculation
"""

import argparse
import os
import signal
import sys

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

###############################################################################
# Setup: Logger, Exception Handling (Exit Signal), and Global Seeds
###############################################################################
sys.path.append("/workspace/dso/gensar/dift/src")
# available log levels: debug, info, success, warning, error, critical, section
from utils.logger import CustomLogger
from utils.error_handler import excepthook, signal_handler
from utils.dota_dataset import LABEL_MAPPING, CLASS_GROUPS


def setup_logger(output_path, log_file_name="linprob.log"):
    return CustomLogger("main_logger", output_path, log_file_name)


def setup_error_handling():
    sys.excepthook = excepthook
    signal.signal(signal.SIGINT, signal_handler)


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


###############################################################################
# Feature Dataset and Model Classes
###############################################################################
class FeatureDataset(Dataset):
    def __init__(self, csv_file, root_dir, feature_idx=0):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.feature_idx = feature_idx
        self.mapping = LABEL_MAPPING

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get the full path from the CSV file
        file_path = self.annotations.iloc[index, 0]

        # If root_dir is provided and the path is not absolute, join with root_dir
        if self.root_dir and not os.path.isabs(file_path):
            file_path = os.path.join(self.root_dir, file_path)

        # Replace .png with .pt for feature files
        feature_path = file_path.replace(".png", ".pt")

        # Load the feature tensor and select the specified feature index
        features = torch.load(feature_path, weights_only=True)[
            self.feature_idx
        ]

        # Get the category name from the annotations and convert it to an integer label
        category_name = self.annotations.iloc[index, 1]
        label = self.mapping[category_name][0]
        return features, label


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ResNet18Classifier, self).__init__()
        self.in_channels = input_dim

        self.conv1 = nn.Conv2d(
            input_dim, 1280, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(1280)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(1280, 1, stride=1)
        self.layer2 = self._make_layer(2560, 2, stride=2)
        self.layer3 = self._make_layer(2560, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2560, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


###############################################################################
# Functions: Model Evaluation and Metrics Calculation
###############################################################################
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    return f1, all_preds, all_labels


def calculate_group_f1(true_labels, predictions, group):
    group_indices = [
        LABEL_MAPPING[class_name][0] for class_name in CLASS_GROUPS[group]
    ]
    group_mask = np.isin(true_labels, group_indices)
    group_true = np.array(true_labels)[group_mask]
    group_pred = np.array(predictions)[group_mask]
    return f1_score(group_true, group_pred, average="macro")


def calculate_and_save_metrics(true_labels, predictions, output_dir):
    metrics = classification_report(true_labels, predictions, output_dict=True)
    metrics_df = pd.DataFrame(metrics).transpose()

    # Calculate group-specific metrics
    overall_f1 = f1_score(true_labels, predictions, average="macro")
    head_f1 = calculate_group_f1(true_labels, predictions, "head")
    middle_f1 = calculate_group_f1(true_labels, predictions, "middle")
    tail_f1 = calculate_group_f1(true_labels, predictions, "tail")

    # Remove the last three rows (accuracy, macro avg, weighted avg)
    metrics_df = metrics_df.iloc[:-3]

    # Add new rows for overall, head, middle, and tail metrics
    new_rows = pd.DataFrame(
        {
            "precision": [overall_f1, head_f1, middle_f1, tail_f1],
            "recall": [overall_f1, head_f1, middle_f1, tail_f1],
            "f1-score": [overall_f1, head_f1, middle_f1, tail_f1],
            "support": [len(true_labels)] * 4,
        },
        index=[
            "macro avg (overall)",
            "macro avg (head)",
            "macro avg (middle)",
            "macro avg (tail)",
        ],
    )

    # Concatenate the original metrics with the new rows
    metrics_df = pd.concat([metrics_df, new_rows])

    # Save the updated metrics to CSV
    metrics_file = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_file, index=True)
    logger.success(f"Metrics saved to: {metrics_file}")

    # Log the metrics
    logger.info(f"Overall Macro F1: {overall_f1:.4f}")
    logger.info(f"Head Classes Macro F1: {head_f1:.4f}")
    logger.info(f"Middle Classes Macro F1: {middle_f1:.4f}")
    logger.info(f"Tail Classes Macro F1: {tail_f1:.4f}")

    # Save predictions and labels
    predictions_file = os.path.join(output_dir, "predictions.csv")
    labels_file = os.path.join(output_dir, "labels.csv")

    predictions_df = pd.DataFrame({"predictions": predictions})
    predictions_df.to_csv(predictions_file, index=False)
    logger.success(f"Predictions saved to: {predictions_file}")

    labels_df = pd.DataFrame({"labels": true_labels})
    labels_df.to_csv(labels_file, index=False)
    logger.success(f"Labels saved to: {labels_file}")

    return metrics


###############################################################################
# Main Function
###############################################################################
def main(args):
    global logger
    logger = setup_logger(args.output_dir)
    setup_error_handling()
    setup_seeds(args.random_seed)

    logger.section("Setup of DOTA linear probing")
    logger.info(f"Feature dimension: {args.feat_dim}")
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of classes: {args.class_num}")
    logger.info(f"Input feature directory: {args.feature_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    input_dim = args.feat_dim

    # Load datasets
    logger.section("Loading datasets")
    logger.info(f"Loading train dataset from {args.train_csv}")

    # Use empty string if feature_dir is "/" to avoid double slashes
    feature_dir = "" if args.feature_dir == "/" else args.feature_dir

    train_dataset = FeatureDataset(
        csv_file=args.train_csv,
        root_dir=feature_dir,
        feature_idx=args.feature_idx,
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")

    # Use val_test_dir if provided, otherwise use feature_dir
    val_test_dir = args.val_test_dir or args.feature_dir

    logger.info(f"Loading validation dataset from {args.val_csv}")
    val_dataset = FeatureDataset(
        csv_file=args.val_csv,
        root_dir=val_test_dir,
        feature_idx=args.feature_idx,
    )
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    logger.info(f"Loading test dataset from {args.test_csv}")
    test_dataset = FeatureDataset(
        csv_file=args.test_csv,
        root_dir=val_test_dir,
        feature_idx=args.feature_idx,
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")

    logger.info(f"Creating DataLoaders with {args.dl_workers} workers")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dl_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dl_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dl_workers,
        pin_memory=True,
    )

    # Initialize the output directory
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model
    logger.section("Initializing model")
    logger.info(f"Model architecture: {args.arch}")
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Number of classes: {args.class_num}")
    if args.arch == "linprob":
        model = LinearClassifier(input_dim, args.class_num).cuda()
        logger.info("Using Linear Classifier")
    elif args.arch == "resnet18":
        model = ResNet18Classifier(input_dim, args.class_num).cuda()
        logger.info("Using ResNet18 Classifier")

    model = DataParallel(model)
    logger.info("Model wrapped with DataParallel")
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    logger.info(
        f"Optimizer: SGD with learning rate {args.lr} and momentum 0.9"
    )

    best_val_f1 = 0
    best_epoch = 0

    total_steps = len(train_loader) * args.epochs
    progress_bar = tqdm(total=total_steps, unit="step", desc="Training")

    logger.section("Training")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"Epoch": epoch + 1, "Loss": loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        val_f1, _, _ = evaluate_model(model, val_loader)
        logger.info(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(
                model.state_dict(), os.path.join(output_dir, "best_model.pth")
            )

    progress_bar.close()
    logger.success(
        f"Best Validation F1 Score: {best_val_f1:.4f} at Epoch {best_epoch+1}"
    )

    # Load the best model and evaluate on the test set
    logger.section("Evaluating on test set")
    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, "best_model.pth"), weights_only=True
        )
    )

    test_f1, test_preds, test_labels = evaluate_model(model, test_loader)
    logger.info(f"Test F1-Score: {test_f1}%")

    # Calculate and save metrics
    logger.section("Calculating and saving metrics")
    _ = calculate_and_save_metrics(test_labels, test_preds, output_dir)

    logger.success("DOTA linear probing completed successfully")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a CNN classifier on image features."
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory containing extracted features",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="/workspace/data/DOTA_v2/image_folder/hf_format/train.csv",
        help="Path to the train CSV file",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="/workspace/data/DOTA_v2/image_folder/hf_format/val.csv",
        help="Path to the validation CSV file",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="/workspace/data/DOTA_v2/image_folder/hf_format/val.csv",
        help="Path to the test CSV file",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["linprob", "resnet18"],
        help="Architecture of the model",
    )
    parser.add_argument(
        "--class_num", type=int, default=15, help="Number of classes"
    )
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=1280,
        help="Dimensions of the given features",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, required=True, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the best model",
    )
    parser.add_argument(
        "--feature_idx",
        type=int,
        default=0,
        help="Feature index to use for training",
    )
    parser.add_argument(
        "--dl_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--val_test_dir",
        type=str,
        default=None,
        help="Separate directory for validation and test features (optional)",
    )
    args = parser.parse_args()
    main(args)
