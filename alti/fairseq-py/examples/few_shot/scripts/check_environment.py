if __name__ == "__main__":
    """This script will fail if the environment is correctly set.
        Current checks:
        - torch can be imported
        - fused_layer_norm_cuda is available
        
       Other checks can be added below!
    """
    
    # Make sure that torch is setup
    import torch 
    
    # Make sure that fused_layer_norm_cuda is configured
    import importlib
    fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    

    