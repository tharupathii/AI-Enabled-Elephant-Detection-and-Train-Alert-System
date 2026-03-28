# AI-Enabled-Elephant-Detection-and-Train-Alert-System

uni project

members :
    Tharupathi Bandara CS/2022/038
    Dewmika Senarathna CS/2022/012
    Pubudu Weerasinghe CS/2022/014
    Rumeth Amarasiri CS/2022/039
    Sasindu Dilshara CS/2022/053

1. Install the AI software (This takes about 10 seconds)
!pip install ultralytics -q

from google.colab import files
from ultralytics import YOLO
import glob
import os
from IPython.display import Image, display

1. Upload the picture
print("👇 Click the button below to upload your elephant picture:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

2. THE FIX: Rename .jfif to .jpg automatically
if filename.endswith('.jfif'):
    new_filename = filename.replace('.jfif', '.jpg')
    os.rename(filename, new_filename)
    filename = new_filename
    print(f"🔄 Fixed the file name to: {filename}")

3. Load your newly trained brain
model = YOLO('best_elephant.pt')

4. Tell the AI to look at the picture
print("Checking the image...")
results = model.predict(source=filename, save=True, conf=0.5)

5. Display the results on your screen!
latest_prediction = max(glob.glob('runs/detect/predict*/*.jpg'))
display(Image(filename=latest_prediction))
