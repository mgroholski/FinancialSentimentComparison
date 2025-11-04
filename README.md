# Financial Sentiment Analysis Project
## About
Within the project we provide a pipeline to run financial sentiment analysis on four different machine learning methods: convolutional neural network, long short-term memory, random forest classifier, and support vector machine.

## Dataset
We use data from the (FNSPID dataset)[https://github.com/Zdong104/FNSPID_Financial_News_Dataset]. We provide processed datasets used during our experimentation for the (full dataset)[https://drive.google.com/file/d/1UK-OwzI7j0ITMmF1IDKxxZPrneJP9x3m/view?usp=sharing] and a (fortune 500 subset)[https://drive.google.com/file/d/1tBKFjc_ilOJ3La_Kd9--UURZBTTvqfO0/view?usp=share_link].

## Run
1. Download dependencies with `pip3 install -r requirements.txt`.
2. Ensure dataset is downloaded from above section.
3. Run `python3 main.py -r 2 -d ./data_utils/fortune_500 -p svm`
	We elabortate on command line arugments below.

## Command Line Argument
`-r` or `--runs`: An integer representing amount of repeats to test the data on a pipeline.
`-d` or `--data`: The path to the data folder.
`-p` or `--pipeline`: A string representing which model to use. Choices for the argument are: ["svm", "cnn", "lstm", "randomforest", "all"]
