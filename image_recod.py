# image_recog.py
# V14
"""
        self.conn.execute("set memory_limit = '12GB'")
        self.conn.execute("set max_memory = '12GB'")
"""
"""
Image Recognition System with Performance Tracking and Model Fine-tuning
Analyzes images from O:\Bilder\1-D7100\2015-06-23-Bonmont
Added this instead O:\Bilder\1-D7100\2015-06-24-LaBarillette
Stores results in DuckDB and uses corrections to fine-tune the model
"""

import os
import time
import psutil
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
import gc
import shutil

class CorrectionDataset(Dataset):
    """Dataset for fine-tuning with user corrections"""
    def __init__(self, corrections_data, processor):
        self.data = corrections_data
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        
        # Process image and text with explicit truncation
        encoding = self.processor(
            images=image,
            text=item['correction'],
            padding="max_length",
            max_length=50,
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return encoding

class ImageRecognitionSystem:
    def __init__(self, image_folder, db_path="images.db", model_save_path="fine_tuned_model"):
        self.image_folder = Path(image_folder)
        self.db_path = db_path
        self.model_save_path = model_save_path
        self.conn = None
        
        # Cost constants (EUR)
        self.CPU_COST_PER_SEC = 1.0
        self.IO_COST = 0.1
        self.ELAPSED_COST_PER_SEC = 0.001
        
        # Performance tracking
        self.process = psutil.Process()
        
        # Model will be loaded/initialized later
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_or_initialize_model(self):
        """Load fine-tuned model if exists, otherwise load base model"""
        model_path = Path(self.model_save_path)
        
        if model_path.exists() and (model_path / "config.json").exists():
            print(f"Loading fine-tuned model from {self.model_save_path}...")
            self.processor = BlipProcessor.from_pretrained(self.model_save_path)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_save_path)
            print("✓ Fine-tuned model loaded successfully")
        else:
            print("Loading base BLIP model...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            print("✓ Base model loaded")
        
        self.model.to(self.device)
        print(f"Model running on {self.device}")
        
    def init_database(self):
        """Initialize DuckDB database with schema"""
        self.conn = duckdb.connect(self.db_path)
        self.conn.execute("set memory_limit = '12GB'")
        self.conn.execute("set max_memory = '12GB'")
        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS image_iterations (
                id INTEGER PRIMARY KEY,
                image_path VARCHAR,
                image_name VARCHAR,
                iteration INTEGER,
                interpretation TEXT,
                accuracy_score INTEGER,
                user_correction TEXT,
                cpu_time_sec FLOAT,
                elapsed_time_sec FLOAT,
                io_count INTEGER,
                cpu_cost_eur FLOAT,
                io_cost_eur FLOAT,
                elapsed_cost_eur FLOAT,
                total_cost_eur FLOAT,
                timestamp TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS image_summary (
                image_name VARCHAR PRIMARY KEY,
                total_iterations INTEGER,
                final_interpretation TEXT,
                final_accuracy INTEGER,
                all_corrections TEXT,
                total_cpu_time_sec FLOAT,
                total_elapsed_time_sec FLOAT,
                total_io_count INTEGER,
                total_cost_eur FLOAT,
                avg_accuracy FLOAT,
                timestamp TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_training_history (
                id INTEGER PRIMARY KEY,
                training_date TIMESTAMP,
                num_corrections_used INTEGER,
                training_epochs INTEGER,
                training_loss FLOAT,
                training_time_sec FLOAT,
                model_path VARCHAR
            )
        """)
        
        print("Database initialized successfully")
        
    def load_corrections_from_db(self) -> List[Dict]:
        """Load all user corrections from database"""
        query = """
            SELECT DISTINCT
                image_path,
                image_name,
                user_correction,
                accuracy_score
            FROM image_iterations
            WHERE user_correction IS NOT NULL 
            AND user_correction != ''
            AND accuracy_score < 80
            ORDER BY timestamp DESC
        """
        
        results = self.conn.execute(query).fetchall()
        
        corrections = []
        for row in results:
            corrections.append({
                'image_path': row[0],
                'image_name': row[1],
                'correction': row[2],
                'accuracy': row[3]
            })
        
        return corrections
    
    def fine_tune_model(self, epochs=3, batch_size=2, learning_rate=5e-5):
        """Fine-tune model using corrections from database"""
        print("\n" + "="*60)
        print("FINE-TUNING MODEL WITH USER CORRECTIONS")
        print("="*60)
        
        # Load corrections
        corrections = self.load_corrections_from_db()
        
        if not corrections:
            print("No corrections found in database. Skipping fine-tuning.")
            return False
        
        print(f"Found {len(corrections)} corrections for training")
        
        # Display sample corrections
        print("\nSample corrections:")
        for i, corr in enumerate(corrections[:3]):
            print(f"  {i+1}. {corr['image_name']}: {corr['correction']}")
        
        # Create dataset
        dataset = CorrectionDataset(corrections, self.processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare model for training
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        print(f"\nTraining for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"✓ Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        print(f"  Average training loss: {avg_loss:.4f}")
        
        # Clean up memory before saving
        print("\nPreparing to save model...")
        del optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Ensure save directory exists and is accessible
        save_path = Path(self.model_save_path)
        
        # If directory exists, remove it to avoid file lock issues
        if save_path.exists():
            print(f"Removing existing model directory: {save_path}")
            try:
                shutil.rmtree(save_path)
            except Exception as e:
                print(f"Warning: Could not remove existing directory: {e}")
                print("Attempting to save anyway...")
        
        # Create fresh directory
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save fine-tuned model with error handling
        print(f"Saving fine-tuned model to {self.model_save_path}...")
        try:
            self.model.save_pretrained(
                self.model_save_path,
                safe_serialization=True,
                max_shard_size="5GB"
            )
            self.processor.save_pretrained(self.model_save_path)
            print("✓ Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
            print("Attempting alternative save method...")
            try:
                # Try saving with safe_serialization=False
                self.model.save_pretrained(
                    self.model_save_path,
                    safe_serialization=False
                )
                self.processor.save_pretrained(self.model_save_path)
                print("✓ Model saved successfully (using PyTorch format)")
            except Exception as e2:
                print(f"Failed to save model: {e2}")
                return False
        
        # Record training in database
        self.conn.execute("""
            INSERT INTO model_training_history (
                id, training_date, num_corrections_used, training_epochs,
                training_loss, training_time_sec, model_path
            ) VALUES (
                (SELECT COALESCE(MAX(id), 0) + 1 FROM model_training_history),
                ?, ?, ?, ?, ?, ?
            )
        """, [
            datetime.now(), len(corrections), epochs,
            avg_loss, training_time, str(self.model_save_path)
        ])
        
        return True
    
    def get_io_counters(self):
        """Get current I/O counters"""
        io = self.process.io_counters()
        return io.read_count + io.write_count
    
    def interpret_image(self, image_path, previous_correction=""):
        """Generate interpretation for an image"""
        # Track I/O before
        io_before = self.get_io_counters()
        cpu_time_before = self.process.cpu_times().user + self.process.cpu_times().system
        elapsed_before = time.time()
        
        # Load and process image
        img = Image.open(image_path).convert('RGB')
        
        # Generate caption with optional text guidance
        if previous_correction:
            text_prompt = f"a photo of {previous_correction}"
            inputs = self.processor(
                img, 
                text=text_prompt, 
                return_tensors="pt",
                max_length=50,
                truncation=True
            ).to(self.device)
        else:
            inputs = self.processor(
                img, 
                return_tensors="pt",
                max_length=50,
                truncation=True
            ).to(self.device)
        
        # Set model to eval mode for inference
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)
        interpretation = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Track I/O after
        elapsed_time = time.time() - elapsed_before
        cpu_time_after = self.process.cpu_times().user + self.process.cpu_times().system
        cpu_time = cpu_time_after - cpu_time_before
        io_after = self.get_io_counters()
        io_count = io_after - io_before
        
        return {
            'interpretation': interpretation,
            'cpu_time': cpu_time,
            'elapsed_time': elapsed_time,
            'io_count': io_count,
            'image': img
        }
    
    def calculate_costs(self, cpu_time, io_count, elapsed_time):
        """Calculate costs based on resource usage"""
        cpu_cost = cpu_time * self.CPU_COST_PER_SEC
        io_cost = io_count * self.IO_COST
        elapsed_cost = elapsed_time * self.ELAPSED_COST_PER_SEC
        total_cost = cpu_cost + io_cost + elapsed_cost
        
        return {
            'cpu_cost': cpu_cost,
            'io_cost': io_cost,
            'elapsed_cost': elapsed_cost,
            'total_cost': total_cost
        }
    
    def display_image_with_interpretation(self, img, interpretation, image_name, iteration):
        """Display image with interpretation"""
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{image_name} - Iteration {iteration}\n{interpretation}", 
                 fontsize=12, wrap=True)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def get_accuracy_feedback(self):
        """Solicit accuracy feedback and correction details from user"""
        while True:
            try:
                accuracy = int(input("\nEnter accuracy score (0-100): "))
                if 0 <= accuracy <= 100:
                    break
                else:
                    print("Please enter a value between 0 and 100")
            except ValueError:
                print("Please enter a valid integer")
        
        # Get detailed feedback if accuracy is low
        correction = ""
        if accuracy < 100:
            print("\nPlease describe what is incorrect (or press Enter to skip):")
            print("Example: 'No bench or horse visible. Image contains a woman with a camera.'")
            correction = input("Correction: ").strip()
        
        return accuracy, correction
    
    def process_image(self, image_path):
        """Process a single image with iterative refinement"""
        image_name = image_path.name
        print(f"\n{'='*60}")
        print(f"Processing: {image_name}")
        print(f"{'='*60}")
        
        iteration = 0
        accuracy = 0
        total_cpu_time = 0
        total_elapsed_time = 0
        total_io_count = 0
        total_cost = 0
        accuracies = []
        corrections = []
        final_interpretation = ""
        previous_correction = ""
        
        while accuracy < 50:
            iteration += 1
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration}:")
            print(f"{'─'*60}")
            
            # Interpret image (with guidance from previous correction if available)
            result = self.interpret_image(image_path, previous_correction)
            
            # Display
            self.display_image_with_interpretation(
                result['image'], 
                result['interpretation'],
                image_name,
                iteration
            )
            
            print(f"Interpretation: {result['interpretation']}")
            print(f"CPU Time: {result['cpu_time']:.4f}s")
            print(f"Elapsed Time: {result['elapsed_time']:.4f}s")
            print(f"I/O Count: {result['io_count']}")
            
            # Get user feedback with correction
            accuracy, correction = self.get_accuracy_feedback()
            
            if correction:
                corrections.append(f"Iteration {iteration}: {correction}")
                previous_correction = correction
                print(f"\n📝 Recorded correction: {correction}")
            
            # Calculate costs
            costs = self.calculate_costs(
                result['cpu_time'],
                result['io_count'],
                result['elapsed_time']
            )
            
            # Accumulate totals
            total_cpu_time += result['cpu_time']
            total_elapsed_time += result['elapsed_time']
            total_io_count += result['io_count']
            total_cost += costs['total_cost']
            accuracies.append(accuracy)
            final_interpretation = result['interpretation']
            
            # Store iteration in database
            self.conn.execute("""
                INSERT INTO image_iterations (
                    id, image_path, image_name, iteration, interpretation, 
                    accuracy_score, user_correction, cpu_time_sec, elapsed_time_sec, 
                    io_count, cpu_cost_eur, io_cost_eur, elapsed_cost_eur, 
                    total_cost_eur, timestamp
                ) VALUES (
                    (SELECT COALESCE(MAX(id), 0) + 1 FROM image_iterations),
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [
                str(image_path), image_name, iteration, result['interpretation'],
                accuracy, correction, result['cpu_time'], result['elapsed_time'],
                result['io_count'], costs['cpu_cost'], costs['io_cost'],
                costs['elapsed_cost'], costs['total_cost'], datetime.now()
            ])
            
            print(f"\nIteration Cost: EUR {costs['total_cost']:.4f}")
            print(f"  - CPU: EUR {costs['cpu_cost']:.4f}")
            print(f"  - I/O: EUR {costs['io_cost']:.4f}")
            print(f"  - Elapsed: EUR {costs['elapsed_cost']:.4f}")
            
            if accuracy >= 50:
                print(f"\n✓ Accuracy threshold met ({accuracy}%)")
                break
            else:
                print(f"\n✗ Accuracy below threshold ({accuracy}% < 50%), refining...")
                if correction:
                    print(f"   Will use correction as guidance for next iteration")
        
        plt.close()
        
        # Store summary
        avg_accuracy = sum(accuracies) / len(accuracies)
        all_corrections_text = " | ".join(corrections) if corrections else ""
        
        self.conn.execute("""
            INSERT OR REPLACE INTO image_summary (
                image_name, total_iterations, final_interpretation, final_accuracy,
                all_corrections, total_cpu_time_sec, total_elapsed_time_sec, 
                total_io_count, total_cost_eur, avg_accuracy, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            image_name, iteration, final_interpretation, accuracy,
            all_corrections_text, total_cpu_time, total_elapsed_time, 
            total_io_count, total_cost, avg_accuracy, datetime.now()
        ])
        
        # Display summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {image_name}")
        print(f"{'='*60}")
        print(f"Total Iterations: {iteration}")
        print(f"Final Accuracy: {accuracy}%")
        print(f"Average Accuracy: {avg_accuracy:.1f}%")
        print(f"Total CPU Time: {total_cpu_time:.4f}s")
        print(f"Total Elapsed Time: {total_elapsed_time:.4f}s")
        print(f"Total I/O Count: {total_io_count}")
        print(f"Total Cost: EUR {total_cost:.4f}")
        print(f"Final Interpretation: {final_interpretation}")
        
        if corrections:
            print(f"\nUser Corrections Provided:")
            for corr in corrections:
                print(f"  • {corr}")
        
        return {
            'image_name': image_name,
            'iterations': iteration,
            'total_cost': total_cost,
            'avg_accuracy': avg_accuracy
        }
    
    def process_all_images(self):
        """Process all JPG images in the folder"""
        jpg_files = list(self.image_folder.glob("*.jpg")) + \
                   list(self.image_folder.glob("*.JPG"))
        
        if not jpg_files:
            print(f"No JPG files found in {self.image_folder}")
            return
        
        print(f"Found {len(jpg_files)} images to process")
        
        results = []
        for img_path in jpg_files:
            result = self.process_image(img_path)
            results.append(result)
        
        return results
    
    def generate_statistics_report(self):
        """Generate graphical report of statistics"""
        # Query summary data
        df = self.conn.execute("""
            SELECT 
                image_name,
                total_iterations,
                final_accuracy,
                total_cpu_time_sec,
                total_elapsed_time_sec,
                total_io_count,
                total_cost_eur,
                avg_accuracy,
                all_corrections
            FROM image_summary
            ORDER BY image_name
        """).fetchdf()
        
        if df.empty:
            print("No data to display")
            return
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Image Recognition Performance Statistics', fontsize=16, fontweight='bold')
        
        # 1. Total Cost per Image
        axes[0, 0].barh(df['image_name'], df['total_cost_eur'], color='steelblue')
        axes[0, 0].set_xlabel('Total Cost (EUR)')
        axes[0, 0].set_title('Total Cost per Image')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Iterations Required
        axes[0, 1].bar(range(len(df)), df['total_iterations'], color='coral')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Iterations')
        axes[0, 1].set_title('Iterations Required per Image')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. CPU Time vs Elapsed Time
        x = range(len(df))
        width = 0.35
        axes[1, 0].bar([i - width/2 for i in x], df['total_cpu_time_sec'], 
                       width, label='CPU Time', color='green', alpha=0.7)
        axes[1, 0].bar([i + width/2 for i in x], df['total_elapsed_time_sec'], 
                       width, label='Elapsed Time', color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('CPU Time vs Elapsed Time')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Accuracy Distribution
        axes[1, 1].scatter(df['avg_accuracy'], df['final_accuracy'], 
                          s=100, alpha=0.6, color='purple')
        axes[1, 1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('Average Accuracy (%)')
        axes[1, 1].set_ylabel('Final Accuracy (%)')
        axes[1, 1].set_title('Average vs Final Accuracy')
        axes[1, 1].grid(alpha=0.3)
        
        # 5. I/O Count per Image
        axes[2, 0].bar(range(len(df)), df['total_io_count'], color='teal')
        axes[2, 0].set_xlabel('Image Index')
        axes[2, 0].set_ylabel('I/O Count')
        axes[2, 0].set_title('Total I/O Operations per Image')
        axes[2, 0].grid(axis='y', alpha=0.3)
        
        # 6. Cost Breakdown and Statistics
        summary_stats = {
            'Total Cost': df['total_cost_eur'].sum(),
            'Avg Cost/Image': df['total_cost_eur'].mean(),
            'Total CPU Time': df['total_cpu_time_sec'].sum(),
            'Total I/O': df['total_io_count'].sum(),
            'Avg Accuracy': df['avg_accuracy'].mean(),
            'Images with Corrections': df['all_corrections'].notna().sum()
        }
        
        axes[2, 1].axis('off')
        summary_text = '\n'.join([f'{k}: {v:.2f}' for k, v in summary_stats.items()])
        axes[2, 1].text(0.1, 0.5, f'Overall Statistics:\n\n{summary_text}',
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        for key, value in summary_stats.items():
            print(f"{key}: {value:.2f}")
        
        # Display correction summary
        corrections_df = df[df['all_corrections'].notna() & (df['all_corrections'] != '')]
        if not corrections_df.empty:
            print("\n" + "="*60)
            print("USER CORRECTIONS SUMMARY")
            print("="*60)
            for idx, row in corrections_df.iterrows():
                print(f"\n{row['image_name']}:")
                print(f"  {row['all_corrections']}")
    
    def display_training_history(self):
        """Display model training history"""
        training_df = self.conn.execute("""
            SELECT 
                training_date,
                num_corrections_used,
                training_epochs,
                training_loss,
                training_time_sec
            FROM model_training_history
            ORDER BY training_date DESC
        """).fetchdf()
        
        if training_df.empty:
            print("\nNo training history found.")
            return
        
        print("\n" + "="*60)
        print("MODEL TRAINING HISTORY")
        print("="*60)
        for idx, row in training_df.iterrows():
            print(f"\nTraining Session {len(training_df) - idx}:")
            print(f"  Date: {row['training_date']}")
            print(f"  Corrections Used: {row['num_corrections_used']}")
            print(f"  Epochs: {row['training_epochs']}")
            print(f"  Final Loss: {row['training_loss']:.4f}")
            print(f"  Training Time: {row['training_time_sec']:.2f}s")
    
    def run(self, mode='process'):
        """
        Main execution flow
        
        mode: 'process' - process images with current model
              'train' - fine-tune model with existing corrections
              'train_and_process' - fine-tune then process images
        """
        self.init_database()
        self.load_or_initialize_model()
        
        if mode in ['train', 'train_and_process']:
            success = self.fine_tune_model()
            if success:
                print("\n" + "="*60)
                print("Reloading fine-tuned model...")
                print("="*60)
                self.load_or_initialize_model()
        
        if mode in ['process', 'train_and_process']:
            self.process_all_images()
            self.generate_statistics_report()
        
        self.display_training_history()
        
        if self.conn:
            self.conn.close()
        print("\n✓ Processing complete. Results stored in images.db")


# Main execution
if __name__ == "__main__":
    import sys
    
    # Initialize system
    system = ImageRecognitionSystem(
#        image_folder=r"O:\Bilder\1-D7100\2015-06-23-Bonmont",
        image_folder=r"O:\Bilder\1-D7100\2015-06-24-LaBarillette",
        db_path="images.db",
        model_save_path="fine_tuned_model"
    )
    
    # Determine mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nSelect mode:")
        print("1. Process images (use existing model)")
        print("2. Train model (fine-tune with corrections from DB)")
        print("3. Train and Process (fine-tune then process images)")
        choice = input("\nEnter choice (1-3): ").strip()
        
        mode_map = {'1': 'process', '2': 'train', '3': 'train_and_process'}
        mode = mode_map.get(choice, 'process')
    
    print(f"\n{'='*60}")
    print(f"Running in '{mode}' mode")
    print(f"{'='*60}")
    
    # Run the complete workflow
    system.run(mode=mode)