import urllib.request
import zipfile
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-1m.zip")

def download():
    if os.path.exists(os.path.join(DATA_DIR, "ml-1m")):
        print("Data already downloaded.")
        return
    print("Downloading MovieLens 1M...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    print("Done. Data extracted to data/ml-1m/")

if __name__ == "__main__":
    download()
