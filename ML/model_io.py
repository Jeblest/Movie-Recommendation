import torch
import pickle
import os
from core_model import SimilarContentNN
def save_model(model,filepath):
        if not os.path.exists('model'):
            os.makedirs('model')
        """Saves model state, embeddings, and dataframe."""
        torch.save({
            'bert_model_state_dict': model.bert_model.state_dict(),
            'projection_layer_state_dict': model.projection_layer.state_dict(),
            'type_projection_state_dict': model.type_projection.state_dict(),
            'embeddings_matrix': model.embeddings_matrix,
            'df': model.df
        }, f"model/{filepath}.pt")

        with open(filepath + '_tokenizer.pkl', 'wb') as f:
            pickle.dump(model.tokenizer, f)
        print("Model saved.")

def load_model(model_path):
    """Initialize and load model from checkpoint."""
    # Initialize model
    model = SimilarContentNN(batch_size=32)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=model.device)
        
        # Load model states
        model.model.load_state_dict(checkpoint['model_state_dict'])
        # Load embeddings and dataframe
        model.embeddings_matrix = checkpoint['embeddings_matrix']
        model.df = checkpoint['df']
        # Load tokenizer
        tokenizer_path = model_path + '_tokenizer.pkl'
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                model.tokenizer = pickle.load(f)
       
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None