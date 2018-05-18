Requires Python 2.7

# Instructions

First, install dependencies:

dependencies are found in requirements.txt and can be installed with

```
pip install -r requirements.txt
```
# Data

Next, acquire data:

## SemEval Dataset

For the SemEval2017 data:

The data comes from the Sem Eval competition, and can be found under the **Results** section on the following site:
http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools

For direct download links to the relevant data, please use the following:

The training data can be found here:
https://www.dropbox.com/s/byzr8yoda6bua1b/2017_English_final.zip?dl=1

The testing data can be found here:
http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4-test.zip

The GloVe Twitter Pre-Trained Vectors come from:
https://nlp.stanford.edu/projects/glove/

For a direct download link to the relevant file, please use:
http://nlp.stanford.edu/data/glove.twitter.27B.zip

Please download the SemEval training data, SemEval test data, and GloVe twitter vectors using the direct download links. and place them in the:

data/semeval

folder. Then please unzip all three zip files.

After unzipping, the relevant files can be found in the following locations:

The GloVe vector file to be used is the 200 dimension version:

glove.twitter.27B.200d.txt

The SemEval test data is found in the folder:

semeval2017-task4-test/

and the relevant file is:

* SemEval2017-task4-test.subtask-A.english.txt

The SemEval training data is found in the folder:

2017_English_final/GOLD/Subtask_A

and the relevant files are:

1. twitter-2013dev-A.txt
2. twitter-2013test-A.txt
3. twitter-2013train-A.txt
4. twitter-2014test-A.txt
5. twitter-2015test-A.txt
6. twitter-2015train-A.txt
7. twitter-2016dev-A.txt
8. twitter-2016devtest-A.txt
9. twitter-2016train-A.txt
10. twitter-2016test-A.txt

### Pre-processing

The twitter .txt data needs to be preprocessed using the preprocessing script provided by the GloVe team. I have included this script within the **data/semeval** folder. The name of the file is:

preprocess-twitter.rb

Note that I have made a slight modification to this script, so please do use the version found in the **data/semeval** folder

The script reads stdin and can be invoked from the command line with:

```
ruby -n preprocess-twitter.rb
```

Please output the processed results into the:

data/semeval/processed

folder as follows:

Output the SemEval2017-task4-test.subtask-A.english.txt processed output into processed/test.txt:

```
cat semeval2017-task4-test/SemEval2017-task4-test.subtask-A.english.txt | ruby -n preprocess-twitter.rb > processed/test.txt
```
The resultant output in processed/test.txt should have 12284 lines.

Output the twitter-2016test-A.txt (item 10 in the list above) processed output into processed/twitter-2016test-A.txt:
```
cat 2017_English_final/GOLD/Subtask_A/twitter-2016test-A.txt | ruby -n preprocess-twitter.rb > processed/twitter-2016test-A.txt
```
The resultant output in processed/twitter-2016test-A.txt should have 20633 lines.

Output the processed output of all the other .txt files (items 1-9 in the list above) into processed/aggregate.txt:
```
cat 2017_English_final/GOLD/Subtask_A/twitter-2013dev-A.txt | ruby -n preprocess-twitter.rb > processed/aggregate.txt
cat 2017_English_final/GOLD/Subtask_A/twitter-2013test-A.txt | ruby -n preprocess-twitter.rb >> processed/aggregate.txt
cat 2017_English_final/GOLD/Subtask_A/twitter-2013train-A.txt | ruby -n preprocess-twitter.rb >> processed/aggregate.txt
cat 2017_English_final/GOLD/Subtask_A/twitter-2014test-A.txt | ruby -n preprocess-twitter.rb >> processed/aggregate.txt
cat 2017_English_final/GOLD/Subtask_A/twitter-2015test-A.txt | ruby -n preprocess-twitter.rb >> processed/aggregate.txt
cat 2017_English_final/GOLD/Subtask_A/twitter-2015train-A.txt | ruby -n preprocess-twitter.rb >> processed/aggregate.txt
cat 2017_English_final/GOLD/Subtask_A/twitter-2016dev-A.txt | ruby -n preprocess-twitter.rb >> processed/aggregate.txt
cat 2017_English_final/GOLD/Subtask_A/twitter-2016devtest-A.txt | ruby -n preprocess-twitter.rb >> processed/aggregate.txt
cat 2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt | ruby -n preprocess-twitter.rb >> processed/aggregate.txt
```
The resultant output in processed/aggregate.txt should have 29616 lines.

The GloVe vectors need to be converted into a different format in order to be read by the gensim python module I am using. You can make this conversion by running:
```
python -m gensim.scripts.glove2word2vec --input  glove.twitter.27B.200d.txt --output glove.twitter.27B.200d.w2vformat.txt
```
from the command line. Please note that you must already have installed the required dependencies (as detailed above) in order to access the gensim module required for this.

For convenience, I have included some shell scripts within the **data/semeval** folder to perform the downloading and processing described above. You may need to first `chmod +x` them to make them executable.

If you wish to use them, please open a terminal and navigate to the **data/semeval** folder. You can then run them from the command line as follows:

To download and unzip the SemEval data:
```
./downloadAndUnzipSemEvalData
```
To process the SemEval data:
```
./processSemEvalData
```
To download and unzip the GloVe vectors:
```
./downloadAndUnzipGloveVectors
```
To process the GloVe vectors:
```
./processGloveVectors
```
At the end of all this, the resultant filesystem structure should be as follows:

* sent760/
  * data/
    * semeval/
      * 2017_English_final/
        * DOWNLOAD/...
        * GOLD/
          * README.txt
          * Subtask_A/
            * README.txt
            * livejournal-2014test-A.tsv
            * sms-2013test-A.tsv
            * twitter-2013dev-A.txt
            * twitter-2013test-A.txt
            * twitter-2013train-A.txt
            * twitter-2014sarcasm-A.txt
            * twitter-2014test-A.txt
            * twitter-2015test-A.txt
            * twitter-2015train-A.txt
            * twitter-2016dev-A.txt
            * twitter-2016devtest-A.txt
            * twitter-2016test-A.txt
            * twitter-2016train-A.txt
          * Subtasks_BD/
          * Subtasks_CE/
      * 2017_English_final.zip
      * downloadAndUnzipGloveVectors
      * downloadAndUnzipSemEvalData
      * glove.twitter.27B.25d.txt
      * glove.twitter.27B.50d.txt
      * glove.twitter.27B.100d.txt
      * glove.twitter.27B.200d.txt
      * **glove.twitter.27B.200d.w2vformat.txt**
      * glove.twitter.27B.zip
      * preprocess-twitter.rb
      * processed/
        * **aggregate.txt**
        * **test.txt**
        * **twitter-2016test-A.txt**
      * processGloveVectors
      * processSemEvalData
      * SemEval2017-task4-test/
        * SemEval2017-task4-test.subtask-A.arabic.txt
        * SemEval2017-task4-test.subtask-A.english.txt
        * SemEval2017-task4-test.subtask-BD.arabic.txt
        * SemEval2017-task4-test.subtask-BD.english.txt
        * SemEval2017-task4-test.subtask-CE.arabic.txt
        * SemEval2017-task4-test.subtask-CE.english.txt
      * semeval2017-task4-test.zip

The relevant files needed for the models are highlighted in bold.

If you have any trouble with the preprocessing of the data, I have included the processed data here:
https://www.dropbox.com/sh/xfgav0d4w47yg8p/AAB9qchyH7SCxC5XJEH-NNWea?dl=0

## IMDB Large Movie Review Dataset

For the IMDB movie review data:
