# Write custom code for logging during training
import pandas as pd
import matplotlib.pyplot as plt
import csv

from pathlib import Path
home = Path.home()
print(home)

class Logger():
    def __init__(self, file_name):
        self.file_name = file_name

        self.full_path = home / "downloads/project1/reports/experiment_logs" / file_name

        self.history = []

        self.headers = ['epoch', 'train_loss', 'val_loss', 'accuracy', 'f1_score', 'lr']
        
        
        with open(self.full_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            
        print(f"Logger initialized. File created at: {self.full_path}")
    def log(self, metrics):
        self.history.append(metrics)
        
        # 2. Atomic write to CSV (protects data if training crashes)
        with open(self.file_name, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(metrics)
            
        print(f"Epoch {metrics['epoch']} logged successfully.")
    
    def visualise(self):
        if not self.history:
            print("No data to visualize yet.")
            return

        df = pd.DataFrame(self.history)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves (essential for detecting overfitting)
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='green', marker='o')
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='orange', marker='o')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        #Accuracy/F1 Change
        ax2.plot(df['epoch'], df['accuracy'], label='Accuracy', color='green', marker='s')
        ax2.plot(df['epoch'], df['f1_score'], label='F1 Score', color='orange', marker='s')
        ax2.set_title('Metric Performance')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def end(self):
        if not self.history:
            return None
            
        # Find the row with the highest f1_score
        best_epoch_data = max(self.history, key=lambda x: x['f1_score'])
        
        print("\n--- Final Training Summary ---")
        print(f"Best Epoch: {best_epoch_data['epoch']}")
        print(f"Max F1 Score: {best_epoch_data['f1_score']:.4f}")
        print(f"Final Val Loss: {best_epoch_data['val_loss']:.4f}")
        
        return best_epoch_data

