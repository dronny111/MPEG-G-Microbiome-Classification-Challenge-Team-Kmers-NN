MPEG-G Microbiome Classification Challenge — Team Kmers NN Approach


Overview

Can you classify microbiome samples by body site and individual using compressed data?

Your goal is to build a machine learning model that classifies a microbiome sample’s body site—stool, oral, skin, or nasal—based on its 16S rRNA gene sequence profile encoded in MPEG-G format, along with health status tags, and participant metadata. Briefly, 16S rRNA is used to identify and classify bacteria by sequencing a specific region of the 16S ribosomal RNA gene and aligning it to a database, which is highly conserved among bacteria, allowing for taxonomic classification.

This repository contains the implementation of our neural network solution to the MPEG-G Microbiome Classification Challenge. 

```
## Folder Structure
├── data/                       # Datasets used for training / evaluation ([Data prep process](https://github.com/koleshjr/MPEG-G-Microbiome-Classification-Challenge/tree/main/data_prep)
├── centralised/               # Code for centralized learning setup
├── federated/                 # Code for federated learning setup
└── README.md   
```


Key Components

Feature extraction — [same](https://github.com/koleshjr/MPEG-G-Microbiome-Classification-Challenge/tree/main/data_prep)


To install:

pip install -r requirements.txt

Usage

Extract features and store inside `data/`


Centralised Pipeline:

Training:

chmod +x centralised/run_all.sh


./centralised/run_all.sh


Inference:

python ./centralised/inference.py




Federated Training:


Training:

python federated/run_simulation.py


Inference:

python federated/inference.py
