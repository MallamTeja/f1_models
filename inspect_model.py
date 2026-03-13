import joblib
import sys

path = sys.argv[1]
try:
    artifact = joblib.load(path)
    print(f"Type: {type(artifact)}")
    if isinstance(artifact, dict):
        print(f"Keys: {artifact.keys()}")
        if "imputer" in artifact:
            print(f"Imputer exists: {artifact['imputer'] is not None}")
    else:
        print(f"No dict, object type: {type(artifact)}")
except Exception as e:
    print(f"Error: {e}")
