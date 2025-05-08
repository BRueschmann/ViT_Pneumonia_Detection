#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score, roc_auc_score, confusion_matrix, fbeta_score
from transformers import pipeline, ViTForImageClassification, TrainingArguments, Trainer, ViTFeatureExtractor, EarlyStoppingCallback
import argparse
from scipy.special import softmax # to get probabilities for predicitons and use it for ensemble model
from sklearn.linear_model import LogisticRegression
import seaborn as sns


# 1. Helper functions for Notebook Data Exploration and Preperation
# -------------------------------------------------------------------------------------------------------------------

# Data Exploration

def simple_xray_loader(directory):
    """ Loads all x-ray JPEG images from the specified directory and returns list of images as numpy arrays in grayscale """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg'):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                # Convert image to grayscale
                img_gray = img.convert('L')
                images.append(np.array(img_gray))
    return images

def plot_pixel_intensity(images, bins=100, title="Histogram of Average Pixel Intensities"):
    """ Computes and plots histogram of sum pixel intensities of images. Results normalised so AUC=1"""
    # Concatenate all pixel values from all images
    all_pixels = np.concatenate([image.flatten() for image in images])
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_pixels, bins=bins, color='purple', edgecolor='black', density=True)
    plt.xlabel("Pixel Intensity", fontsize=18)
    plt.ylabel("Normalized Frequency", fontsize=18)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
 
def compare_windowed_images(path1, path2, lower, upper):
    """ Plots two windowed images next to each other. 1 NORMAL, 1 PNEU"""
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    win_img1 = window_image(img1, lower=lower, upper=upper)
    win_img2 = window_image(img2, lower=lower, upper=upper)
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title("Original NORMAL")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(win_img1, cmap='gray')
    axes[0, 1].set_title("Windowed NORMAL")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(img2, cmap='gray')
    axes[1, 0].set_title("Original PNEU")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(win_img2, cmap='gray')
    axes[1, 1].set_title("Windowed PNEU")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.show()

# Balancing train and val

def group_files_by_person(file_list, class_name):
    """
    Group list of filenames by person ID
    For PNEUMONIA the person ID is everything before the first '_'
    For NORMAL each filename is unique
    """
    groups = {}
    for filename in file_list:
        # pneu takes everyything before '_' which is something like 'Person1'
        person_id = filename.split('_')[0] if class_name.upper() == 'PNEUMONIA' else filename
        groups.setdefault(person_id, []).append(filename)
    return groups

def move_files(train_dir, val_dir, file_list):
    """ Move all files in file_list from train_dir to val_dir """
    moved = 0
    for filename in file_list:
        src = os.path.join(train_dir, filename)
        dst = os.path.join(val_dir, filename)
        shutil.move(src, dst)
        moved += 1
    return moved

def balance_class_val(train_dir, val_dir, target_val_count, class_name):
    """ Balance train and val set for one class (NORMAL or PNEUMONIA) by moving whole person groups from train_dir to val_dir """
    files = [f for f in os.listdir(train_dir) if f.lower().endswith('.jpeg')]
    print(f"Total images in {train_dir}: {len(files)}")
    
    # Group files by person
    groups = group_files_by_person(files, class_name)
    groups_list = list(groups.items())
    random.shuffle(groups_list)
    
    # moves images until target count is reached
    moved_count = 0
    for person_id, file_list in groups_list:
        if moved_count >= target_val_count:
            break
        moved_count += move_files(train_dir, val_dir, file_list)
    
    # Confirm results
    new_train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    new_val_count = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    print(f"After moving for {class_name}: train has {new_train_count} images, val {class_name} has {new_val_count} images.")


# Evaluation Functions. I should turn them into 1 or modularise when i have time.


def evaluate_model(pipe, dataloader, classes):
    """ 
    Evaluate a model pipeline 
    Returns: "loss", "accuracy", "precision", "recall", "f1", "auc", "confusion_matrix"
    """
    all_true = []
    all_preds = []
    all_probs = []  # probability for the positive class
    losses = []


    # Create a transform to convert tensor images back to PIL images
    to_pil = transforms.ToPILImage()
    
    for images, labels in dataloader:
        # Convert each tensor image in the batch to a PIL image
        pil_images = [to_pil(img) for img in images]
        # Get predictions from the pipeline (each output is a list of dicts for each image)
        outputs = pipe(pil_images)
        
        # Looping over all images, using zip to pair each output with its true label
        for out, true_label in zip(outputs, labels):
            # Build a probability vector (assumes two classes), 
            # to form a full probability distribution for each image
            # Make list (prob_vector) of zeros with a length equal to number of classes (NORMAL, PNEUMONIA)
            # This list will store the probability for each class
            prob_vector = [0.0] * len(classes)
            for item in out:
                # The model returns a dict with keys "label" and "score"
                # extract predicted class name (cls_name) and its probability (score) from dictionary
                # And store them in correct position in prob_vector
                cls_name = item['label']
                score = item['score']
                if cls_name.startswith("LABEL_"):
                    # Extract the numeric part and convert to int (e.g. "LABEL_1" becomes 1)
                    idx = int(cls_name.split("_")[-1])
                else:
                    idx = classes.index(cls_name)
                prob_vector[idx] = score
            # Convert list to NumPy array for easier math
            prob_vector = np.array(prob_vector)
            # Max in prob vector is final predicted class for the image (pred_label)
            pred_label = int(np.argmax(prob_vector))
            
            # Save the predicted probability for the positive class in index 1, needed for AUC
            all_probs.append(prob_vector[1])
            all_true.append(int(true_label))
            all_preds.append(pred_label)
            # Compute cross-entropy loss for this sample
            # Cross-entropy loss measures how confident the model is about the correct class
            true_prob = prob_vector[int(true_label)]
            loss_i = -np.log(true_prob + 1e-12) # very small addition prevents taking the log of zero
            losses.append(loss_i)
    
    # Computing evaluation metrics
    avg_loss = np.mean(losses)
    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, pos_label=1)
    recall = recall_score(all_true, all_preds, pos_label=1)
    f1 = f1_score(all_true, all_preds, pos_label=1)
    auc = roc_auc_score(all_true, all_probs)
    cm = confusion_matrix(all_true, all_preds)
    
    # printing evaluation metrics
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return {"loss": avg_loss, "accuracy": accuracy, "precision": precision, 
            "recall": recall, "f1": f1, "auc": auc, "confusion_matrix": cm}


# Evaluation function for simple models

def evaluate_simple_model(model, X, y, title=None):
    """
    Evaluates a classification model:
      - Prints accuracy, precision, recall, F1 and F2 scores
      - Plots the ROC curve and normalized (rate-based) confusion matrix
    """
    # Predict class labels
    y_prediction = model.predict(X)
    
    # Try to obtain probability estimates for ROC curve
    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_probability = model.decision_function(X)
    else:
        y_probability = None

    # Calculate metrics
    accuracy = accuracy_score(y, y_prediction)
    precision = precision_score(y, y_prediction)
    recall = recall_score(y, y_prediction)
    f1 = f1_score(y, y_prediction)
    f2 = fbeta_score(y, y_prediction, beta=2)
    
    # Print evaluation metrics
    print(f"{title} Performance:")
    print("Accuracy: ", f"{accuracy:.3f}")
    print("Precision:", f"{precision:.3f}")
    print("Recall:   ", f"{recall:.3f}")
    print("F1 Score: ", f"{f1:.3f}")
    print("F2 Score: ", f"{f2:.3f}\n")
    
    # Plot normalized confusion matrix (row-wise percentages)
    plt.figure(figsize=(6, 5))
    cm_norm = confusion_matrix(y, y_prediction, normalize='true')
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greys")
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Plot ROC curve if probability estimates are available
    if y_probability is not None:
        fpr, tpr, _ = roc_curve(y, y_probability)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.3f})", color='purple')
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title} ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("No probability estimates available for ROC curve.")


# 2. Helper functions for Main training function
# -------------------------------------------------------------------------------------------------------------------

# Had to define higher level rgb function and class to fix error in training
# It wouldn't take rgb conversion inside training funciton
def to_rgb(x):
    return x.convert("RGB")

class WindowTransform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    def __call__(self, x):
        return window_image(x, lower=self.lower, upper=self.upper).convert("RGB")

# I know people window with centre and width, but lower and upper makes more sense to me
def window_image(pil_img, lower=30, upper=200):
    """Clamps pixel intensities in the range [lower, upper] and rescales to [0, 255]."""
    # Convert to grayscale to be safe
    pil_img = pil_img.convert('L')
    # Convert to NumPy array
    np_img = np.array(pil_img, dtype=np.float32)
    # Clamp intensities
    np_img = np.clip(np_img, lower, upper)
    # Rescale to [0, 1]
    np_img = (np_img - lower) / float(upper - lower)
    # Convert to [0, 255] for an 8-bit image
    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    # Convert back to PIL
    return Image.fromarray(np_img)

def create_transforms(phase, windowing=False, lower=30, upper=200):
    """Returns a composed set of transforms for the given phase with augmentation for training."""
    transform_list = [transforms.Resize((224, 224))]
    
    # Augmentation parameters from Manickam et al (2021)
    # Except gaussian blurr sigma cause 0 gave error
    if phase.lower() == "train":
        # Data augmentation: 50% chance to horizontally flip
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        # 50% chance to vertically flip
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        # Random affine translation: up to 12 pixels shift on a 224x224 image (~0.0536 fraction)
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(12/224, 12/224)))
        # Gaussian blur with sigma up to 1
        transform_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 1.0)))
    
    if windowing:
        transform_list.append(WindowTransform(lower, upper))
    else:
        transform_list.append(transforms.Lambda(to_rgb))
    
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def create_dataset(root_dir, phase, windowing=False, lower=30, upper=200):
    """Creates an ImageFolder dataset for a given phase eg. train """
    phase_dir = os.path.join(root_dir, phase)
    transform = create_transforms(phase, windowing=windowing, lower=lower, upper=upper)
    dataset = datasets.ImageFolder(phase_dir, transform=transform)
    return dataset

def create_dataloader(dataset, batch_size=32, shuffle=False, num_workers=4):
    """Wraps dataset in DataLoader"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def xray_loader(root_dir, batch_size=32, num_workers=4, windowing=False, lower=30, upper=200):
    """Loads X-ray data from root_dir and returns (data_loaders, datasets_dict)"""
    phases = ['train', 'val', 'test']
    datasets_dict = {}
    data_loaders = {}
    # for each phase create transformed data set
    for phase in phases:
        dataset = create_dataset(root_dir, phase, windowing=windowing, lower=lower, upper=upper)
        datasets_dict[phase] = dataset
        # shuffle train set and not val and test
        shuffle = True if phase == 'train' else False
        data_loaders[phase] = create_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loaders, datasets_dict

# When there is time, this function should be modularised or combined with earlier evaluation functions

def compute_metrics(eval_pred):
    """
    Compute and print evaluation metrics during training, 
    With continuous probabilities using softmax to convert raw logits into probabilities before computing predictions
    """
    logits, labels = eval_pred
    # Convert logits to probabilities using softmax so that outputs are between 0 and 1
    probs = softmax(logits, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, pos_label=1)
    recall = recall_score(labels, predictions, pos_label=1)
    f1 = f1_score(labels, predictions, pos_label=1)
    auc = roc_auc_score(labels, probs[:, 1])
    cm = confusion_matrix(labels, predictions).tolist()
    
    metrics = {
        "accuracy": acc, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "auc": auc, 
        "confusion_matrix": cm
    }
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # include probabilities
    metrics["probabilities"] = probs.tolist()  
    return metrics

# Had to make custom collator to fix error
def dictionary_data_collator(features):
    """Converts a list of (image, label) tuples to a dictionary for the Trainer"""
    pixel_values = [f[0] for f in features]
    labels = [f[1] for f in features]
    return {"pixel_values": torch.stack(pixel_values), "labels": torch.tensor(labels)}


# 3. Main training function
# -------------------------------------------------------------------------------------------------------------------

def train_windowing_model(model_name, training_args, root_dir, batch_size, num_workers=4, windowing=False, lower=30, upper=200, output_dir="."):
    # Set manual seed, same as Nicholas
    torch.manual_seed(42)
    
    # Load feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    # Load data
    data_loaders, datasets_dict = xray_loader(
        root_dir=root_dir, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        windowing=windowing, 
        lower=lower, 
        upper=upper
    )
    
    # Load pre-trained model with 2 output labels
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)
    
    # Ensure the training arguments output_dir is set
    training_args.output_dir = output_dir
    
    # Create Trainer with Early Stopping to avoid overfitting (spoiler: it overfit anyways)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets_dict['train'],
        eval_dataset=datasets_dict['val'],
        compute_metrics=compute_metrics,
        processing_class=feature_extractor,
        data_collator=dictionary_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model and track training loss
    trainer.train()
    
    # Save the model (and all other important stuff) to output_dir
    trainer.save_model(output_dir)
    
    # Evaluate on the validation set
    results = trainer.evaluate()
    
    # If available plot training loss from log history 
    if hasattr(trainer.state, "log_history"):
        loss_logs = [log for log in trainer.state.log_history if "loss" in log]
        if loss_logs:
            steps = [log["step"] for log in loss_logs if "loss" in log]
            losses = [log["loss"] for log in loss_logs if "loss" in log]
            plt.figure(figsize=(8, 6))
            plt.plot(steps, losses, marker='o', color='purple')
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            loss_plot_path = os.path.join(output_dir, "training_loss.png")
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Training loss plot saved to {loss_plot_path}")
    
    return results, model, trainer



# 4. Functions for Ensemble Model
# -------------------------------------------------------------------------------------------------------------------

def create_trainer_for_model(root_dir, model_dir, phase, batch_size=16, num_workers=4,
                             windowing=False, lower=30, upper=200):
    """To load data, model, feature extractor, and create Trainer for a model """

    data_loaders, datasets_dict = xray_loader(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        windowing=windowing,
        lower=lower,
        upper=upper
    )
    
    model = ViTForImageClassification.from_pretrained(model_dir)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
    
    trainer = Trainer(
        model=model,
        eval_dataset=datasets_dict[phase],   # or use "train" if that’s what you need
        compute_metrics=compute_metrics,
        processing_class=feature_extractor,
        data_collator=dictionary_data_collator
    )
    return trainer, datasets_dict


def get_continuous_predictions(trainer, dataset):
    """
    Runs predictions using the trainer on the provided dataset and returns
    the probability outputs (continuous scores between 0 and 1 for each class).
    """
    predictions_output = trainer.predict(dataset)
    logits = predictions_output.predictions
    # Convert logits to probabilities with softmax
    probs = softmax(logits, axis=-1)
    return probs


# 5. Functions for Prediction Video
# -------------------------------------------------------------------------------------------------------------------

def get_all_image_paths(directory):
    """Return a list of all file paths in 'NORMAL' first, then 'PNEUMONIA', excluding the folders themselves."""
    paths = []
    for folder_name in ['NORMAL', 'PNEUMONIA']:
        folder_path = os.path.join(directory, folder_name)
        files = glob.glob(os.path.join(folder_path, '**', '*'), recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        paths.extend(files)
    return paths

def add_image_paths_to_df(df, base_dir, phase):
    """ Adds and fills image_paths column to DF """
    # Construct the directory path for this phase
    phase_dir = os.path.join(base_dir, phase)
    print("Looking in:", phase_dir)

    image_paths = get_all_image_paths(phase_dir)

    if len(image_paths) != len(df):
        print("Ooops: Number of images = ({}), and number of rows in DF = ({})."
              .format(len(image_paths), len(df)))

    # Add image paths to the DataFrame
    df = df.copy()
    df['image_path'] = image_paths[:len(df)]  # Avoid index error if mismatch

    return df

def get_frame_size(df, image_col='image_path'):
    """ Determines the frame size by reading the first image in the DataFrame """
    first_image_path = df[image_col].iloc[0]
    frame = cv2.imread(first_image_path)
    if frame is None:
        raise ValueError(f"Could not load the first image: {first_image_path}")
    height, width, channels = frame.shape
    print(f"Determined frame size: {(width, height)}")
    return (width, height)

def setup_video_writer(output_filename, fps, frame_size):
    """ Sets up the OpenCV video writer with the specified output, frame rate, and frame size """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    return video_writer

def process_frame(row, frame_size, font_scale=5, thickness=7):
    """
    Reads an image, resizes if necessary, and overlays the prediction and true label.
    Text is displayed in green if the prediction is correct, or red if not.
    """
    image_path = row['image_path']
    image = cv2.imread(image_path)

    # Resize the image to the desired frame size if it doesn't match
    if (image.shape[1], image.shape[0]) != frame_size:
        image = cv2.resize(image, frame_size)
    
    # Set up font and calculate overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlay_text = f"Pred: {row['pred']}, True: {row['target']}"
    
    # Determine text colour: green if correct, red if incorrect.
    text_colour = (0, 255, 0) if row['pred'] == row['target'] else (0, 0, 255)
    
    # Get text size to create a background box.
    (text_width, text_height), baseline = cv2.getTextSize(overlay_text, font, font_scale, thickness)
    margin = 10 
    # Compute text position and overlay the text.
    text_position = (20 + margin, 20 + text_height + margin)
    cv2.putText(image, overlay_text, text_position, font, font_scale, text_colour, thickness, cv2.LINE_AA)

    return image

def create_video(df, output_video, fps=3, image_col='image_path'):
    """ Main function to create prediction video from DF """
    
    # set up frame size and video writer
    frame_size = get_frame_size(df, image_col)
    video_writer = setup_video_writer(output_video, fps, frame_size)
    
    # go through each image
    for idx, row in df.iterrows():
        print(f"Processing frame {idx}: {row['image_path']}")
        frame = process_frame(row, frame_size)
        if frame is not None:
            video_writer.write(frame)
    
    # Save video
    video_writer.release()
    print("Video saved as:", output_video)



# 6. Main function to run script to train a model with windowed data
# -------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a ViT model for chest X-ray pneumonia classification with optional windowing."
    )
    # Parse arguments from slurm script
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k", help="Pre-trained model name")
    parser.add_argument("--windowing", action="store_true", help="Enable windowing")
    parser.add_argument("--lower", type=int, default=30, help="Lower window bound")
    parser.add_argument("--upper", type=int, default=200, help="Upper window bound")
    parser.add_argument("--data_root", type=str, default="chest_xray", help="Root directory for data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs (model, logs, plots)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    args = parser.parse_args()
    
    # Define training arguments like Nicholas but added save_strategy, eval_strategy, and load_best_model_at_end
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=2e-05,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        lr_scheduler_type="linear",
        seed=42,
        logging_steps=10,
        disable_tqdm=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        eval_strategy="epoch",   # Evaluation at the end of each epoch
        save_strategy="epoch",   # Checkpoints saved at end of each epoch
        load_best_model_at_end=True,  # Automatically load best model at end
        metric_for_best_model="eval_loss",
        greater_is_better=False      # For loss, lower is better
    )
    
    # To be safe, create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model with the specified parameters
    results, model, trainer = train_windowing_model(
        model_name=args.model_name,
        training_args=training_args,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        windowing=args.windowing,
        lower=args.lower,
        upper=args.upper,
        output_dir=args.output_dir
    )
    
    print("Final Evaluation Results:", results)

if __name__ == "__main__":
    main()

