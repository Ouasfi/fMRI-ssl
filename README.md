# FMRI-ssl
Matching representation between FMRI and Audio data with self supervised learning
# Setup 
1. Git clone:
```
git clone 
cd FMRI-ssl
```
2. Create a venv and install requirements 
```
python3 -m venv ./.venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. Run tests
````
python test.py
````
# Data
1. install datalad:
````
pip install datalad
````
2. Install the rep
````
datalad -r install ///labs/hasson/narratives/derivatives/fmriprep
````
2. load stimuli
````
datalad install -r -g ///labs/hasson/narratives/stimuli
````
3. load parcellations in the parcellatino dir
# Usage
-  The training process is described in 'Fmri-audio.ipynb`
