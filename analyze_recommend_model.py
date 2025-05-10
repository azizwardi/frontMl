import pickle
import sys
import os

# Increase recursion limit for complex objects
sys.setrecursionlimit(10000)

def analyze_model():
    try:
        # Try to load the model
        model_path = os.path.join('models', 'Recommendmodel.pkl')
        print(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            # Read the first few bytes to check the file format
            header = f.read(10)
            f.seek(0)  # Reset file pointer to beginning
            
            print(f"File header (hex): {header.hex()}")
            
            # Try to load the model
            model = pickle.load(f)
            
            print(f"Model loaded successfully!")
            print(f"Model type: {type(model)}")
            
            # Try to get model attributes
            if hasattr(model, '__dict__'):
                print(f"Model attributes: {model.__dict__.keys()}")
            else:
                print(f"Model doesn't have __dict__ attribute")
                
            # If model is a dictionary
            if isinstance(model, dict):
                print(f"Model is a dictionary with keys: {model.keys()}")
                
            # If model is a list
            elif isinstance(model, list):
                print(f"Model is a list with {len(model)} elements")
                if len(model) > 0:
                    print(f"First element type: {type(model[0])}")
                    
            # Try to call predict method if it exists
            if hasattr(model, 'predict'):
                print("Model has predict method")
            else:
                print("Model doesn't have predict method")
                
            return model
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    model = analyze_model()
