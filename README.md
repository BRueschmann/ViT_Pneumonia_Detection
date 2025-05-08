# ViT_Pneumonia_Detection
This repository contains the files and code for the University of St Andrews Module ID5220 - Medical Imaging and Sensing - Practical 2: Medical Image Analysis. The goal is to use deep learning to diagnose pneumonia from chest x-rays. This project attempted to replicate the best performing open-source model (vision transformer). Then it explored the effects of windowing in data preprocessing and attempted to improve diagnostic performance by fine-tuning vision transformers with different windowing parameters, and training a second stage classifier on their predictions. The project further includes a comparative analysis (table) of different imaging modalities as well as a video of the developed model's predictions.

Directory Structure:
* notebook_mi_p2.ipynb   (Jupyter notebook displaying all tasks)
* code_mi_p2.py   (Python script used to train models on HPC and containing all functions used by the notebook)
* report_mi_p2.pdf.   (pdf report explaining all steps and results in detail)
* slurm_script_mi_p2.sh  (Example Bash script used to submit jobs to HPC)
* video_mi_p2.mp4  (Video of predictions of final produced ensemble model)
* Requirements.txt   (Python packages needed to run the code)
