import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import datasets.datasets as datasets
import model_functions
import matplotlib.pyplot as plt
from our_models import pcam_teacher, waterbirds_teacher
import utils
import os
# Device setup
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def calculate_agreement(student, teacher, dataloader):
    """
    Calculate agreement between student and teacher predictions.
    """
    student.eval()
    teacher.eval()
    total = 0
    agree = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Agreement"):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images, labels = images.to(device), labels.to(device)

            student_logits = student(images)
            teacher_logits = teacher(images)

            # Predictions
            student_preds = student_logits.argmax(dim=1)
            teacher_preds = teacher_logits.argmax(dim=1)

            # Agreement calculation
            agree += (student_preds == teacher_preds).sum().item()
            total += labels.size(0)

    return agree / total

def plot_metrics_live(metrics, epochs_completed, save_path="metrics.png"):
    """
    Plot top-1 and top-5 accuracies and agreement metrics live after each epoch.
    """
    plt.figure(figsize=(15, 5))
    plt.clf()  # Clear the figure
    
    # Plot Top-1 and Top-5 Accuracies
    plt.subplot(1, 3, 1)
    plt.plot(
        [i + 1 for i, v in enumerate(metrics["train_top1_acc"]) if v is not None],
        [v for v in metrics["train_top1_acc"] if v is not None],
        label="Train Top-1 Acc"
    )
    plt.plot(
        [i + 1 for i, v in enumerate(metrics["val_top1_acc"]) if v is not None],
        [v for v in metrics["val_top1_acc"] if v is not None],
        label="Val Top-1 Acc"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Top-1 Accuracy")
    
    plt.subplot(1, 3, 2)
    plt.plot(
        [i + 1 for i, v in enumerate(metrics["train_top5_acc"]) if v is not None],
        [v for v in metrics["train_top5_acc"] if v is not None],
        label="Train Top-5 Acc"
    )
    plt.plot(
        [i + 1 for i, v in enumerate(metrics["val_top5_acc"]) if v is not None],
        [v for v in metrics["val_top5_acc"] if v is not None],
        label="Val Top-5 Acc"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Top-5 Accuracy")
    
    # Plot Agreement with Teacher if available
    plt.subplot(1, 3, 3)
    if metrics["agreement"]:
        plt.plot(
            [i + 1 for i, v in enumerate(metrics["agreement"]) if v is not None],
            [v for v in metrics["agreement"] if v is not None],
            label="Agreement"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Agreement")
        plt.legend()
        plt.title("Agreement with Teacher")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.pause(0.01)  # Pause to refresh the plot

def plot_gradcam_heatmaps(student, teacher, dataloader, gradcam_student, gradcam_teacher, epoch):
    """
    Generate and save Grad-CAM heatmaps for a few samples.
    """
    student.eval()
    teacher.eval()

    samples, labels, _ = next(iter(dataloader))
    images = samples.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        # Generate heatmaps
        student_heatmaps,_ = gradcam_student.generate_heatmap(images, device)
        teacher_heatmaps,_ = gradcam_teacher.generate_heatmap(images, device)

    # Save a few samples
    for i in range(min(5, len(images))):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axs[0].set_title("Input Image")

        axs[1].imshow(student_heatmaps[i].cpu().numpy(), cmap="jet")
        axs[1].set_title("Student Heatmap")

        axs[2].imshow(teacher_heatmaps[i].cpu().numpy(), cmap="jet")
        axs[2].set_title("Teacher Heatmap")

        for ax in axs:
            ax.axis("off")

        save_path = f"heatmaps_epoch_{epoch}_sample_{i}.png"
        plt.savefig(save_path)
        plt.close()

def train_epoch(student, teacher, dataloader, optimizer, scheduler, num_classes, gradcam_student=None, gradcam_teacher=None, temperature=4.0, lambda_weight=5, experiment="e2KD"):
    """
    Train for one epoch using the specified experiment type (baseline, kd, or e2KD).
    """
    student.train()
    if teacher:
        teacher.eval()
    scaler = GradScaler()
    running_loss = 0.0
    
    # Initialize metrics
    top1_metric = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    if num_classes == 2:
        top5_metric = MulticlassAccuracy(num_classes=num_classes, top_k=2).to(device)
    else:
        top5_metric = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)

    for batch in tqdm(dataloader, desc="Training"):
        if len(batch) == 3:  # Imagenet has explanations whilst waterbirds do not
            images, labels, _ = batch
        else:
            images, labels = batch

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision context
        with autocast():

            if experiment == "baseline":
                student_logits = student(images)
                loss = nn.CrossEntropyLoss()(student_logits, labels)
            elif experiment == "kd":
                with torch.no_grad():
                    teacher_logits = teacher(images)
                student_logits = student(images)
                kd_loss = model_functions.kd_loss(student_logits, teacher_logits, temperature)
                gt_loss = nn.CrossEntropyLoss()(student_logits, labels)
                loss = 2*kd_loss + 1*gt_loss
                #loss = kd_loss
            elif experiment == "e2KD":
                teacher_explanations, teacher_logits = gradcam_teacher.generate_heatmap(images, device, retain=False)
                student_explanations, student_logits = gradcam_student.generate_heatmap(images, device, retain=True)
                
                teacher_explanations = teacher_explanations.detach()
                teacher_logits = teacher_logits.detach()

                #student_explanations = student_explanations.detach()

                if False: # as before, did no work using generate_heatmap
                    student_logits = student(images)
                    with torch.no_grad():
                        teacher_logits = teacher(images)

                e2kd_loss = model_functions.e2KD_loss(student_logits, teacher_logits, student_explanations, teacher_explanations, temperature, lambda_weight)
                loss = e2kd_loss 
            else:
                raise ValueError(f"Unknown experiment type: {experiment}")

        # Scale loss and backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

        # Step optimizer
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        
       # Update metrics
        with torch.no_grad():
            top1_metric.update(student_logits, labels)
            top5_metric.update(student_logits, labels)

    # Compute metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    top1_accuracy = top1_metric.compute().item()
    top5_accuracy = top5_metric.compute().item()

    # Update the learning rate scheduler
    scheduler.step()
    
    return epoch_loss, top1_accuracy, top5_accuracy

def evaluate(model, dataloader, num_classes):
    """
    Evaluate the model on validation/test data.
    """
    model.eval()
    device = next(model.parameters()).device  # Ensure compatibility with the model's device

    # Initialize TorchMetrics metrics for Top-1 and Top-5 accuracy
    top1_metric = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    if num_classes == 2:
        top5_metric = MulticlassAccuracy(num_classes=num_classes, top_k=2).to(device)
    else:
        top5_metric = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Update metrics
            top1_metric.update(outputs, labels)
            top5_metric.update(outputs, labels)

    # Compute final metrics
    top1_accuracy = top1_metric.compute().item()
    top5_accuracy = top5_metric.compute().item()

    return top1_accuracy, top5_accuracy

def main(args):
    #global device
    #device = f'cuda:{args.device}' 
    #device = f'cuda' 
    print(f"Using device: {device}")
    # Load datasets
    if args.dataset == 'waterbirds':
        has_id_ood = True
        num_classes = 2
    elif args.dataset == 'imagenet':
        has_id_ood = False
        num_classes = 1000
    elif args.dataset == 'camelyon':
        has_id_ood = False
        num_classes = 2
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    

    global EXPERIMENT_PATH 
    EXPERIMENT_PATH = f"/home/shared_project/dl-adv-group11/src/experiments/{args.dataset}/{args.nr}/"
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    metrics_checkpoint_path = f"{args.dataset}_{args.experiment}_metrics_checkpoint.pth"
    fig_path = f"{args.dataset}_{args.experiment}_plot_metrics.png"
    dataloaders = datasets.get_dataloaders(args.dataset)

    student = models.resnet18(pretrained=False).to(device)
    student.fc = nn.Linear(student.fc.in_features, num_classes).to(device)
    
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(0))
        student.load_state_dict(checkpoint)
        start_epoch = utils.path2num(args.resume)
        print(f"Starting from epoch: {start_epoch}")
        try:
            metrics = torch.load(EXPERIMENT_PATH+metrics_checkpoint_path)
            print("Loaded metrics from checkpoint.")
        except FileNotFoundError:
            print("Metrics checkpoint not found. Initializing empty metrics.")
            metrics = {
                "train_top1_acc": [None] * start_epoch,  # Placeholder for missed epochs
                "train_top5_acc": [None] * start_epoch,
                "val_top1_acc": [None] * start_epoch,
                "val_top5_acc": [None] * start_epoch,
                "agreement": [None] * start_epoch
            }
    else:
        start_epoch = 0
        metrics = {
            "train_top1_acc": [],
            "train_top5_acc": [],
            "val_top1_acc": [],
            "val_top5_acc": [],
            "agreement": []
        }


    teacher = None
    gradcam_teacher = gradcam_student = None
    if args.experiment in ["kd", "e2KD"]:
        if args.dataset == "imagenet":
            teacher = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
            #teacher.fc = nn.Linear(teacher.fc.in_features, num_classes).to(device)
        elif args.dataset == "camelyon":
            teacher = pcam_teacher()
            teacher = teacher.to(device)
        elif args.dataset == "waterbirds":
            teacher = waterbirds_teacher()#.to(torch.device('cuda:0'))
        teacher.eval()  # Freeze teacher weights
        if args.experiment == "e2KD":
            gradcam_teacher = model_functions.GradCAM(teacher, target_layer="layer4")
            gradcam_student = model_functions.GradCAM(student, target_layer="layer4")

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01, weight_decay=1e-4)

    tunable_params = [p for p in student.parameters() if p.requires_grad]
    # print("Tunable Parameters:")
    # for param in tunable_params:
    #     print(param.size())

    scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / 5),  # Warmup for 5 epochs
        CosineAnnealingLR(optimizer, T_max=args.epochs) 
    ],
    milestones=[5]
    )

    # Adjust the scheduler state to resume correctly
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
        
    # Early Stopping Variables
    val_top1_acc = -1.0
    best_val_top1_acc = 0.0  # Initialize the best validation top-1 accuracy
    patience = 500            # Number of epochs to wait after last improvement
    epochs_no_improve = 0    # Counter for epochs since last improvement
    early_stop = False       # Flag to indicate whether to stop training

    # Initialize live plotting
    plt.ion()  # Turn on interactive mode

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if early_stop:
            print("Early stopping triggered. Stopping training.")
            break
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_top1_acc, train_top5_acc = train_epoch(
            student, teacher, dataloaders[args.set], optimizer, scheduler, num_classes,
            gradcam_student=gradcam_student, gradcam_teacher=gradcam_teacher,
            temperature=args.temperature, lambda_weight=args.lambda_weight,
            experiment=args.experiment
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Top-1 Acc: {train_top1_acc:.4f}, Train Top-5 Acc: {train_top5_acc:.4f}")
        metrics["train_top1_acc"].append(train_top1_acc)
        metrics["train_top5_acc"].append(train_top5_acc)

        if teacher:
            if has_id_ood:
                agreement_id = calculate_agreement(student, teacher, dataloaders['val_id'])
                print(f"In-Distribution Agreement with Teacher: {agreement_id:.4f}")

                agreement_ood = calculate_agreement(student, teacher, dataloaders['val_ood'])
                print(f"Out-of-Distribution Agreement with Teacher: {agreement_ood:.4f}")

                # Append average agreement for plotting
                avg_agreement = (agreement_id + agreement_ood) / 2
                metrics["agreement"].append(avg_agreement)
            else:
                # Generic agreement calculation
                agreement = calculate_agreement(student, teacher, dataloaders['val'])
                print(f"Agreement with Teacher: {agreement:.4f}")
                metrics["agreement"].append(agreement)
        else:
            # Append a placeholder if teacher is not used
            metrics["agreement"].append(0.0)

        # Evaluate for Waterbirds (ID/OOD) or default behavior
        metrics["epoch"].append(val_top5_acc)
        if has_id_ood:
            id_top1_acc, id_top5_acc = evaluate(student, dataloaders['val_id'], num_classes)
            ood_top1_acc, ood_top5_acc = evaluate(student, dataloaders['val_ood'], num_classes)
            print(f"In-Distribution Top-1 Accuracy: {id_top1_acc:.4f}, Top-5 Accuracy: {id_top5_acc:.4f}")
            print(f"Out-of-Distribution Top-1 Accuracy: {ood_top1_acc:.4f}, Top-5 Accuracy: {ood_top5_acc:.4f}")
            metrics["val_top1_acc"].append(id_top1_acc)  # Use ID accuracy as val accuracy
            metrics["val_top5_acc"].append(id_top5_acc)
            val_top1_acc =id_top1_acc
        else:
            val_top1_acc, val_top5_acc = evaluate(student, dataloaders['val'], num_classes)
            print(f"Validation Top-1 Accuracy: {val_top1_acc:.4f}, Top-5 Accuracy: {val_top5_acc:.4f}")
            metrics["val_top1_acc"].append(val_top1_acc)
            metrics["val_top5_acc"].append(val_top5_acc)

        plot_metrics_live(metrics, epoch + 1, save_path=EXPERIMENT_PATH+fig_path)
        torch.save(metrics, EXPERIMENT_PATH+metrics_checkpoint_path)
        # Early Stopping Check
    
        if val_top1_acc > best_val_top1_acc:
            best_val_top1_acc = val_top1_acc
            epochs_no_improve = 0

            # Save the best model
            best_model_path = EXPERIMENT_PATH+f"best_student_model_{args.dataset}_{args.experiment}.pth"
            torch.save(student.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Validation accuracy did not improve for {patience} epochs. Early stopping.")
            early_stop = True

        # Save Grad-CAM heatmaps every 15 epochs
        if args.experiment == "e2KD" and epoch % 15 == 0:
            pass#plot_gradcam_heatmaps(student, teacher, dataloaders['val'], gradcam_student, gradcam_teacher, epoch + 1)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = EXPERIMENT_PATH+f"student_model_{args.dataset}_{args.experiment}_epoch_{epoch + 1}.pth"
            torch.save(student.state_dict(), model_path)
            print(f"Model saved: {model_path}")

    plt.ioff()  

    # Test evaluation
    if has_id_ood:
        test_id_top1_acc, test_id_top5_acc = evaluate(student, dataloaders['test_id'], num_classes)
        test_ood_top1_acc, test_ood_top5_acc = evaluate(student, dataloaders['test_ood'], num_classes)
        print(f"Test In-Distribution Top-1 Accuracy: {test_id_top1_acc:.4f}, Top-5 Accuracy: {test_id_top5_acc:.4f}")
        print(f"Test Out-of-Distribution Top-1 Accuracy: {test_ood_top1_acc:.4f}, Top-5 Accuracy: {test_ood_top5_acc:.4f}")
        agreement_id = calculate_agreement(student, teacher, dataloaders['test_id'])
        print(f"In-Distribution Agreement with Teacher: {agreement_id:.4f}")

        agreement_ood = calculate_agreement(student, teacher, dataloaders['test_ood'])
        print(f"Out-of-Distribution Agreement with Teacher: {agreement_ood:.4f}")
    else:
        test_top1_acc, test_top5_acc = evaluate(student, dataloaders['test'], num_classes)
        print(f"Test Top-1 Accuracy: {test_top1_acc:.4f}, Top-5 Accuracy: {test_top5_acc:.4f}")

        agreement = calculate_agreement(student, teacher, dataloaders['test'])
        print(f"Agreement with Teacher: {agreement:.4f}")

    # Save the final model
    if args.epochs !=1:
        model_name = EXPERIMENT_PATH+f"student_model_{args.dataset}_{args.experiment}.pth"
        torch.save(student.state_dict(), model_name)
        print(f"Final model saved as '{model_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for baseline, KD, and e2KD experiments.")
    parser.add_argument("--dataset", required=True, choices=["imagenet", "waterbirds", "camelyon"], help="Dataset to use.")
    parser.add_argument("--experiment", required=True, choices=["baseline", "kd", "e2KD"], help="Type of experiment to run.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--temperature", type=float, default=5.0, help="Temperature for KD loss.")
    parser.add_argument("--lambda_weight", type=float, default=5, help="Weight for explanation loss in e2KD.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a saved model to resume training.")
    parser.add_argument("--device", type=int, default=0, help="Device 0,1,2,3")
    parser.add_argument("--set", type=str, default='train', help="Set to train on")
    parser.add_argument("--nr", type=int, default=0, help="Experiment number")
    args = parser.parse_args()

    print(torch.get_num_threads())

    main(args)