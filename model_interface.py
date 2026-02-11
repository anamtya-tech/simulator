"""
Streamlit interface for model training and management.

This module provides:
1. Dataset selection and management
2. Model training with hyperparameter configuration
3. Model evaluation and visualization
4. Model deployment for inference
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import torch
from dataset_manager import DatasetManager
from model_trainer import ModelTrainer
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class ModelInterface:
    """Streamlit interface for model training"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.dataset_manager = DatasetManager(output_dir)
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session state
        if 'trainer' not in st.session_state:
            st.session_state.trainer = ModelTrainer(self.models_dir)
        if 'training_complete' not in st.session_state:
            st.session_state.training_complete = False
    
    def render(self):
        """Render the model training interface"""
        st.subheader("🤖 Model Training & Management")
        st.markdown("Train a lightweight CNN for audio source classification")
        
        # Tabs for different functions
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dataset Management",
            "🎓 Train Model", 
            "📈 Evaluate Model",
            "🚀 Deploy Model"
        ])
        
        with tab1:
            self._render_dataset_management()
        
        with tab2:
            self._render_training()
        
        with tab3:
            self._render_evaluation()
        
        with tab4:
            self._render_deployment()
    
    def _render_dataset_management(self):
        """Render dataset management interface"""
        st.markdown("### Dataset Management")
        
        # Current configuration
        config = self.dataset_manager._load_config()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Dataset", config['active_dataset'])
        with col2:
            st.metric("Total Datasets", len(config['datasets']))
        
        # List all datasets
        st.markdown("#### Available Datasets")
        datasets = self.dataset_manager.list_datasets()
        
        if not datasets:
            st.info("No datasets found. Run the analyzer to create datasets.")
            return
        
        # Display dataset info
        dataset_info = []
        for dataset_name in datasets:
            stats = self.dataset_manager.get_dataset_stats(dataset_name)
            if stats:
                dataset_info.append({
                    'Name': stats['name'],
                    'Samples': stats['total_samples'],
                    'Classes': len(stats['label_distribution']),
                    'Runs': stats['runs_processed'],
                    'Created': stats['created_at'][:10] if stats.get('created_at') else 'N/A',
                    'Updated': stats['last_updated'][:10] if stats.get('last_updated') else 'N/A'
                })
        
        if dataset_info:
            st.dataframe(dataset_info, width='stretch')
        
        # Dataset actions
        st.markdown("#### Dataset Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Create new dataset
            new_dataset_name = st.text_input(
                "New Dataset Name (leave empty for auto-naming)",
                placeholder=f"{config['dataset_base_name']}X"
            )
            if st.button("➕ Create New Dataset"):
                if new_dataset_name:
                    name, path = self.dataset_manager.set_active_dataset(new_dataset_name)
                else:
                    name, path = self.dataset_manager.create_new_dataset()
                st.success(f"Created dataset: {name}")
                st.rerun()
        
        with col2:
            # Set active dataset
            selected_dataset = st.selectbox(
                "Select Active Dataset",
                datasets,
                index=datasets.index(config['active_dataset']) if config['active_dataset'] in datasets else 0
            )
            if st.button("✓ Set Active"):
                self.dataset_manager.set_active_dataset(selected_dataset)
                st.success(f"Active dataset: {selected_dataset}")
                st.rerun()
        
        with col3:
            # Configure confidence threshold
            new_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, config['confidence_threshold'],
                help="Minimum confidence to save samples"
            )
            if st.button("💾 Update Threshold"):
                config['confidence_threshold'] = new_threshold
                self.dataset_manager._save_config(config)
                st.success(f"Threshold updated to {new_threshold:.2f}")
        
        # Show detailed stats for selected dataset
        st.markdown("#### Dataset Details")
        selected_for_detail = st.selectbox(
            "View Dataset Details",
            datasets,
            key="detail_selector"
        )
        
        if selected_for_detail:
            stats = self.dataset_manager.get_dataset_stats(selected_for_detail)
            if stats:
                st.json(stats)
                
                # Load and show sample distribution
                df = self.dataset_manager.load_dataset(selected_for_detail, include_metadata=True)
                if not df.empty:
                    st.markdown("##### Label Distribution")
                    label_counts = df['label'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    label_counts.plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_xlabel('Label')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Sample Distribution in {selected_for_detail}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show sample preview
                    with st.expander("📄 Sample Preview"):
                        st.dataframe(df.head(10))
    
    def _render_training(self):
        """Render model training interface"""
        st.markdown("### Train Model")
        
        # Check if datasets exist
        datasets = self.dataset_manager.list_datasets()
        if not datasets:
            st.warning("No datasets available. Please create datasets first.")
            return
        
        # Select dataset for training
        col1, col2 = st.columns(2)
        with col1:
            selected_dataset = st.selectbox(
                "Select Dataset for Training",
                datasets,
                key="train_dataset_selector"
            )
        
        with col2:
            # Show dataset stats
            stats = self.dataset_manager.get_dataset_stats(selected_dataset)
            if stats:
                st.metric("Total Samples", stats['total_samples'])
                st.metric("Classes", len(stats['label_distribution']))
        
        # Load dataset
        if st.button("📂 Load Dataset"):
            with st.spinner("Loading dataset..."):
                df = self.dataset_manager.load_dataset(selected_dataset, include_metadata=False)
                
                if df.empty:
                    st.error("Dataset is empty!")
                    return
                
                st.session_state.training_df = df
                st.success(f"Loaded {len(df)} samples with {df['label'].nunique()} classes")
                st.session_state.training_complete = False
        
        if 'training_df' not in st.session_state:
            st.info("Click 'Load Dataset' to begin")
            return
        
        df = st.session_state.training_df
        
        # Training configuration
        st.markdown("#### Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_epochs = st.number_input("Epochs", 10, 200, 50, step=10)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        
        with col2:
            test_size = st.slider("Validation Split", 0.1, 0.4, 0.2, step=0.05)
            patience = st.number_input("Early Stopping Patience", 5, 50, 10)
        
        with col3:
            random_seed = st.number_input("Random Seed", 1, 1000, 42)
            load_existing = st.checkbox("Continue from existing model", value=False)
        
        # Load existing model if requested
        if load_existing:
            existing_models = list(self.models_dir.glob("*.pth"))
            if existing_models:
                selected_model = st.selectbox(
                    "Select Model to Continue",
                    existing_models,
                    format_func=lambda x: x.name
                )
                if st.button("📥 Load Model"):
                    try:
                        st.session_state.trainer.load_checkpoint(selected_model.name)
                        st.success(f"Loaded model: {selected_model.name}")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
            else:
                st.info("No existing models found")
        
        # Train button
        if st.button("🎓 Start Training", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Prepare data
                    train_loader, val_loader = st.session_state.trainer.prepare_data(
                        df, test_size=test_size, random_state=random_seed
                    )
                    
                    # Create model if not loaded
                    if st.session_state.trainer.model is None:
                        num_classes = df['label'].nunique()
                        # Auto-detect input size from data
                        bin_cols = [col for col in df.columns if col.startswith('bin_')]
                        input_size = len(bin_cols)
                        st.info(f"Creating model with input_size={input_size} bins")
                        st.session_state.trainer.create_model(num_classes, input_size=input_size)
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Train
                    history = st.session_state.trainer.train(
                        train_loader, val_loader,
                        num_epochs=num_epochs,
                        lr=learning_rate,
                        patience=patience,
                        save_best=True
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Training complete!")
                    
                    # Save final model
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.trainer.save_checkpoint(
                        f"model_{selected_dataset}_{timestamp}.pth"
                    )
                    
                    # Store results
                    st.session_state.training_history = history
                    st.session_state.training_complete = True
                    
                    st.success("✅ Training completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Show training results
        if st.session_state.training_complete and 'training_history' in st.session_state:
            st.markdown("#### Training Results")
            
            history = st.session_state.training_history
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Train Accuracy", f"{history['train_acc'][-1]:.2f}%")
                st.metric("Best Train Loss", f"{min(history['train_loss']):.4f}")
            with col2:
                st.metric("Final Val Accuracy", f"{history['val_acc'][-1]:.2f}%")
                st.metric("Best Val Loss", f"{min(history['val_loss']):.4f}")
            
            # Plot training curves
            fig = st.session_state.trainer.plot_training_history(history)
            st.pyplot(fig)
    
    def _render_evaluation(self):
        """Render model evaluation interface"""
        st.markdown("### Evaluate Model")
        
        # Load model
        existing_models = list(self.models_dir.glob("*.pth"))
        if not existing_models:
            st.warning("No trained models found. Train a model first.")
            return
        
        selected_model = st.selectbox(
            "Select Model to Evaluate",
            existing_models,
            format_func=lambda x: x.name,
            key="eval_model_selector"
        )
        
        # Load model metadata
        metadata_file = self.models_dir / f"{selected_model.stem}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classes", metadata['num_classes'])
            with col2:
                st.metric("Parameters", f"{metadata['parameter_count']:,}")
            with col3:
                val_acc = metadata.get('val_acc') or metadata.get('best_val_acc', 0)
                if val_acc is None:
                    val_acc = 0
                st.metric("Val Accuracy", f"{val_acc:.2f}%")
            
            st.json(metadata)
        
        # Load model
        if st.button("📥 Load Model for Evaluation"):
            try:
                st.session_state.trainer.load_checkpoint(selected_model.name)
                st.success(f"Loaded model: {selected_model.name}")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return
        
        if st.session_state.trainer.model is None:
            st.info("Load a model to evaluate")
            return
        
        # Select dataset for evaluation
        datasets = self.dataset_manager.list_datasets()
        if not datasets:
            st.warning("No datasets available for evaluation")
            return
        
        eval_dataset = st.selectbox(
            "Select Dataset for Evaluation",
            datasets,
            key="eval_dataset_selector"
        )
        
        if st.button("🔍 Run Evaluation"):
            with st.spinner("Evaluating model..."):
                try:
                    # Load dataset
                    df = self.dataset_manager.load_dataset(eval_dataset, include_metadata=False)
                    
                    if df.empty:
                        st.error("Dataset is empty!")
                        return
                    
                    # Prepare data
                    bin_cols = [col for col in df.columns if col.startswith('bin_')]
                    bin_cols = sorted(bin_cols, key=lambda x: int(x.split('_')[1]))
                    X = df[bin_cols].values
                    y_true = df['label'].values
                    
                    # Predict
                    y_pred, confidences = st.session_state.trainer.predict(X)
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='weighted'
                    )
                    
                    # Display metrics
                    st.markdown("#### Overall Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    with col2:
                        st.metric("Precision", f"{precision:.3f}")
                    with col3:
                        st.metric("Recall", f"{recall:.3f}")
                    with col4:
                        st.metric("F1 Score", f"{f1:.3f}")
                    
                    # Confusion matrix
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=st.session_state.trainer.label_encoder.classes_,
                               yticklabels=st.session_state.trainer.label_encoder.classes_,
                               ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    ax.set_title('Confusion Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Classification report
                    st.markdown("#### Classification Report")
                    report = classification_report(y_true, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format(precision=3))
                    
                    # Confidence distribution
                    st.markdown("#### Confidence Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(confidences, bins=50, color='steelblue', edgecolor='black')
                    ax.set_xlabel('Confidence')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Prediction Confidence Distribution')
                    ax.axvline(0.85, color='red', linestyle='--', label='Threshold (0.85)')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Low confidence samples
                    low_conf_threshold = 0.85
                    low_conf_mask = confidences < low_conf_threshold
                    low_conf_count = low_conf_mask.sum()
                    
                    st.markdown(f"#### Low Confidence Predictions (< {low_conf_threshold})")
                    st.metric("Count", low_conf_count)
                    st.metric("Percentage", f"{100*low_conf_count/len(confidences):.1f}%")
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    def _render_deployment(self):
        """Render model deployment interface"""
        st.markdown("### Deploy Model")
        
        # Load model
        existing_models = list(self.models_dir.glob("*.pth"))
        if not existing_models:
            st.warning("No trained models found. Train a model first.")
            return
        
        selected_model = st.selectbox(
            "Select Model to Deploy",
            existing_models,
            format_func=lambda x: x.name,
            key="deploy_model_selector"
        )
        
        # Show model info
        metadata_file = self.models_dir / f"{selected_model.stem}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            st.markdown("#### Model Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classes", metadata['num_classes'])
                st.write("**Classes:**", metadata['label_encoder'])
            with col2:
                st.metric("Parameters", f"{metadata['parameter_count']:,}")
                file_size_mb = selected_model.stat().st_size / (1024 * 1024)
                st.metric("Model Size", f"{file_size_mb:.2f} MB")
            with col3:
                if metadata.get('val_acc'):
                    st.metric("Validation Acc", f"{metadata['val_acc']:.2f}%")
                st.write("**Trained:**", metadata['timestamp'][:10])
        
        # Deployment options
        st.markdown("#### Deployment Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✓ Set as Active Model"):
                # Copy to active model location
                active_model_path = self.models_dir / 'active_model.pth'
                import shutil
                shutil.copy(selected_model, active_model_path)
                
                # Also copy metadata
                if metadata_file.exists():
                    shutil.copy(metadata_file, self.models_dir / 'active_model_metadata.json')
                
                st.success(f"✅ Deployed {selected_model.name} as active model")
                st.info("The analyzer will now use this model for predictions")
        
        with col2:
            if st.button("📦 Export for Raspberry Pi"):
                st.info("Exporting model for deployment...")
                
                # Load and optimize model for inference
                try:
                    trainer = ModelTrainer(self.models_dir)
                    trainer.load_checkpoint(selected_model.name)
                    
                    # Save optimized version
                    export_path = self.models_dir / f"{selected_model.stem}_optimized.pth"
                    
                    # Use TorchScript for better performance
                    trainer.model.eval()
                    example_input = torch.randn(1, 1024)
                    traced_model = torch.jit.trace(trainer.model, example_input)
                    torch.jit.save(traced_model, export_path)
                    
                    st.success(f"✅ Exported to {export_path.name}")
                    st.download_button(
                        "📥 Download Optimized Model",
                        data=export_path.read_bytes(),
                        file_name=export_path.name,
                        mime="application/octet-stream"
                    )
                    
                except Exception as e:
                    st.error(f"Error exporting model: {e}")
        
        # Show current active model
        st.markdown("---")
        st.markdown("#### Current Active Model")
        active_model_path = self.models_dir / 'active_model.pth'
        if active_model_path.exists():
            st.success(f"Active model exists: {active_model_path.name}")
            
            metadata_path = self.models_dir / 'active_model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    active_metadata = json.load(f)
                st.json(active_metadata)
        else:
            st.warning("No active model deployed yet")
