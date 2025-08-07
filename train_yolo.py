import argparse
import torch
from ultralytics import YOLO
import logging
import optuna

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_yolo(dataset_yaml, model_type="yolo12x", epochs=10, batch_size=8, img_size=640, hyperparams=None):
    """
    Train YOLOv12 detection model using the prepared dataset and specified hyperparameters.
    
    Args:
        dataset_yaml (str): Path to YOLO dataset YAML file
        model_type (str): YOLO model type (default: yolo12x)
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        img_size (int): Image size for training
        hyperparams (dict): Dictionary of hyperparameters (e.g., lr0, momentum)
    
    Returns:
        tuple: (Path to saved model checkpoint, validation mAP@50), or (None, None) if training fails
    """
    try:
        # Check PyTorch and CUDA status
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize model
        model = YOLO(f"/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/{model_type}_train.pt")  # Load pre-trained YOLOv12 model
        logging.info(f"Loaded model: {model_type}")
        model.to(device)

        # Default hyperparameters if none provided
        training_params = {
            'data': dataset_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': device,
            'project': 'yolo12_detection',
            'name': 'runs',
            'lr0': 0.01,  # Initial learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
        }

        # Update with tuned hyperparameters if provided
        if hyperparams:
            training_params.update(hyperparams)
            logging.info(f"Using hyperparameters: {hyperparams}")

        # Train the model
        results = model.train(**training_params)
        
        # Retrieve validation mAP@50 (or other metric) from results
        # Note: Adjust this based on Ultralytics' results format
        metrics = model.val()
        map50 = metrics.box.map50  # Mean Average Precision at IoU=0.5
        
        logging.info(f"Training completed. Model saved at: {model.ckpt_path}, mAP@50: {map50}")
        return model.ckpt_path, map50
        
    except ImportError:
        logging.error("Required packages not found. Please install ultralytics and optuna:")
        logging.error("pip install ultralytics optuna")
        return None, None
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return None, None

def objective(trial, dataset_yaml, model_type, epochs, batch_size, img_size):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        dataset_yaml (str): Path to dataset YAML
        model_type (str): YOLO model type
        epochs (int): Number of epochs
        batch_size (int): Batch size
        img_size (int): Image size
    
    Returns:
        float: Metric to optimize (mAP@50)
    """
    # Define hyperparameter search space
    hyperparams = {
        'lr0': trial.suggest_float('lr0', 1e-5, 1e-2, log=True),  # Learning rate
        'momentum': trial.suggest_float('momentum', 0.7, 0.999),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 5),
        'warmup_momentum': trial.suggest_float('warmup_momentum', 0.5, 0.95),
        'warmup_bias_lr': trial.suggest_float('warmup_bias_lr', 0.05, 0.2),
    }

    # Train the model with these hyperparameters
    _, map50 = train_yolo(dataset_yaml, model_type, epochs, batch_size, img_size, hyperparams)
    
    # Return the metric to optimize (maximize mAP@50)
    return map50 if map50 is not None else float('-inf')

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv12 detection model with hyperparameter tuning")
    parser.add_argument("--dataset_yaml", type=str, default="/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/data_aic2025/processed/yolo_pose_dataset/dataset.yaml",
                        help="Path to YOLO dataset YAML file")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for training")
    parser.add_argument("--model", type=str, default="yolo12x",
                        help="YOLO model to use (yolo12x, etc.)")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials for hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Create Optuna study to maximize mAP@50
    study = optuna.create_study(direction='maximize')
    logging.info("Starting hyperparameter optimization...")
    study.optimize(lambda trial: objective(trial, args.dataset_yaml, args.model, args.epochs, args.batch_size, args.img_size), n_trials=args.n_trials)
    
    # Log the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    logging.info(f"Best hyperparameters: {best_params}")
    logging.info(f"Best mAP@50: {best_value}")
    
    # Train the model with the best hyperparameters
    logging.info("Training final model with best hyperparameters...")
    model_path, final_map50 = train_yolo(
        dataset_yaml=args.dataset_yaml,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        hyperparams=best_params
    )
    
    if model_path:
        logging.info(f"Final training completed. Model saved at: {model_path}, mAP@50: {final_map50}")
    else:
        logging.error("Final training failed. No model was saved.")

if __name__ == "__main__":
    main()