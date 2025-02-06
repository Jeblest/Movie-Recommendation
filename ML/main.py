import os
from model_io import load_model
from recommender import get_recommendations

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', 'model.pt')
    model = load_model(model_path)
    return model
    
    
if __name__ == '__main__':
    main()