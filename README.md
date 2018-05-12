# Story Cloze NLU

Project contains four parts

### Incorrect Ending Generator
Generating incorrect endings using dependency parse of the training sentences and contradicting them.

### Seq2Seq 
Utilized to generate both incorrect and test correct endings.

### Extraction-Based Prediction 
Extract similar stories from training set and use them to predict correct endings.

### Cogcomp Model
Current State of the art model which is optimized using the incorrect endings generated.
