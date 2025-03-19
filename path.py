from dotenv import load_dotenv
from pathlib import Path
import os
PATH_DOTENV = Path(__file__).parents[0] / ".env"
PATHS = ["DIR_SRC", "DIR_DATA", "LIB_PYNICHE"]

class PathFinder:
    
    def __init__(self):
        self.load_env()
       
    def load_env(self):
        env_exists = load_dotenv(PATH_DOTENV)
        if env_exists:
            print(f"Successfully loaded {PATH_DOTENV}")
        else:
            raise Exception(f"No .env file found in {PATH_DOTENV}")
        self.show()
        
    def find_dirs(self, path):
        return [d for d in path.iterdir() if d.is_dir()]

    def __getitem__(self, key):
        return Path(os.environ.get(key))
    
    def __repr__(self):
        self.show()
        return f"PathFinder: {PATH_DOTENV}"

    def show(self):
        print("Available project environments: ")
        for env in PATHS:
            print(f"    {env}={os.environ[env]}")    
        