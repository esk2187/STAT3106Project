# STAT3106 Project - The Headline Shift
Sophia Berrocal, Samuel DeLong, Annie Dong, Emily Koepp

This repository is for our Final Project in STAT3106 Applied Machine Learning at Columbia University. It includes the following: 

1. **headline_shift_FINAL.ipynb**: Python (.ipynb) script that contains instructions for parsing the headline_shift_NEW scripts and uploading Kagglehub data for uploading and cleaning. This is where we train the model and load graph outputs.
2. **headline_shift_NEW.zip**: zip file containing all necessary files, including:
   - **index.html**: Technical blog post which includes relevant literature review, methods, figures/plots, results/analysis, limitations discussion, references, and appendix. This html is hosted at the link: https://esk2187.github.io/STAT3106Project/headline_shift_NEW/index.html
   - **headline_shift_FINAL.ipynb**: including .ipynb within the zip
   - **outputs** folder: includes all plots showing model performance and inferential applications. Also contains the final dataset of tested headlines with classified ideology.
   - **requirements.txt**: Python package dependencies. Install with pip install -r requirements.txt before running the pipeline locally
   - **app** folder: contains code for active learning app. In our prototype we used this to host a locally-hosted app interface, and crucially we used this to collect our 1,095 labels. For this reason, we keep this original code for documentation purposes.However, our final, permanently-hosted app codebase can be found in the following repository (included separately so as to not interfere with the organization of the current codebase: https://github.com/dingdongtinz/headline-shift/tree/main
   - **data** folder: contains data, including:
      - **qbias_clean**: “QBIAS Media Bias in Search Queries” dataset of 21,000+ headlines labeled left/center/right by political ideology.
   - **emotionality_labels_more**: 1,095 headlines hand-annotated for emotionality training signal.
   - **models** folder: saved model weights for all trained classifiers; generated automatically after running the pipeline. Required for inference without retraining.
   - **src** folder: all Python source modules imported by the pipeline. Includes data loading, model definitions, hyperparameter search, sentiment scoring, inference, and time-series visualization. See individual file docstrings for usage.
   - **run_pipeline.py**: Main entry point for the full end-to-end pipeline. Accepts command-line flags to select model type, number of epochs, emotionality labels path, and whether to skip inference.
   - **README.md**: a Markdown document with detailed instructions for constructing and running data processing, the active learning interface, and the ML pipelines in Colab.
4. **headline_shift_NEW**: included as a directory version of the zip file. Notably, this excludes the following:
   - **Kaggle-sourced data**: We use a Kaggle-sourced dataset with 4.5M headlines across the 10 largest U.S. news sites (2007–2022), from which we filter for CNN, Fox News, The Washington Post, and The New York Times in the 2013–2022 window. This is uploaded directly via JSON.
