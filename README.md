# STAT3106 Project - The Headline Shift
Sophia Berrocal, Samuel DeLong, Annie Dong, Emily Koepp

This repository is for our Final Project in STAT3106 Applied Machine Learning at Columbia University. It includes the following

1. **index.html**: Technical blog post which includes relevant literature review, methods, figures/plots, results/analysis, limitations discussion, references, and appendix. This html is hosted at the link: https://esk2187.github.io/STAT3106Project/index.html
2. **headline_shift_FINAL**: Python (.ipynb) script that contains instructions for parsing the headline_shift_NEW scripts and uploading Kagglehub data for uploading and cleaning
3. **headline_shift_NEW**: included both as a .zip for usage with Colab and a directory containing--
   - **readME.md**: a Markdown document with detailed instructions for constructing and running data processing, the active learning interface, and the ML pipelines in Colab
   - **outputs/** folder: includes all plots showing model performance and inferential applications. Also contains the final dataset of tested headlines with classified ideology.
   - **allsides.csv**: “QBIAS Media Bias in Search Queries” dataset of 21,000+ headlines labeled left/center/right by political ideology.
   - **headlines_clean.csv**: A Kaggle-sourced dataset with 4.5M headlines across the 10 largest U.S. news sites (2007–2022), from which we filter for CNN, Fox News, The Washington Post, and The New York Times in the 2013–2022 window.
   - **Various script files**: labelled, and parsed by the Colab notebook below
   - **!Excluded!**: suggestions.csv (sentimentality training file) excluded from non-compressed version of headling_shift_NEW due to size limits, but exists in the zipped file.
