# E2A9

# Precision, Recall, and F1 Scores

## Definitions

True Positives (TP) : 'True' Predictions given that they belong to the 'True' class

True Negatives (TP) : 'False' Predictions given that they belong to the 'False' class

False Positives (FP) : 'True' Predictions given that they belong to the 'False' class

False Negatives (FP) : 'False' Predictions given that they belong to the 'True' class

<img src="https://render.githubusercontent.com/render/math?math=\text{Precision} = \frac{TP}{TP %2B FP}">

<img src="https://render.githubusercontent.com/render/math?math=\text{Recall} = \frac{TP}{TP  %2B FN}">

<img src="https://render.githubusercontent.com/render/math?math=\text{F1} = 2 * \frac{\text{Precision} \ * \ \text{Recall}}{\text{Precision} %2B \text{Recall}}">

<!--- 
$$\text{Precision} = \frac{TP}{TP + FP}$$

$\text{Recall} = \frac{TP}{TP + FN}$

$\text{F1} = 2 * \frac{\text{Precision} \ * \ \text{Recall}}{\text{Precision} + \text{Recall}}$
-->


## Task

For the demonstration, we have used the notebook from Session 3 but without the addition task. Also, to make the interpretation of the scores easier, we have turned the task into a binary classification problem by categorizing the digits as even or odd. This has been done using the following code

```python
def ev(x):
	if x%2 == 0:
		return 1
	else:
		return 0
	
train_dataset.targets.apply_(ev)
test_dataset.targets.apply_(ev)
```

The number of classes have been reduced to 2 (0 or 1.

# BLEU Scores

It stands for **B**i**L**ingual **E**valuation **U**nderstudy.

It evaluates the quality of machine translation by comparing it with (one or more) ’reference’ translations. It computes two parameters
- clipped precision
- brevity penalty

### Clipped Precision
It is a measure of how much the 'candidate' (prediction) translation matches the 'reference' (actual, human-translated) translation. Steps:
- Consider unigram sequences
- Initialize a counter for the unigrams (for both predicted and actual translation). Note that the counter has unique keys only. All values are counts.
- For each word in the prediction, check if it is in the actual translation.
  - If it is absent, set the value to 0 (in the prediction key)
  - If it is present, ensure that the actual translation’s count is less than or equal to the prediction count. ’Clip’ if needed.
- Sum all the unigram scores of the prediction and divide by the total number of words in the prediction.
- Repeat this for bigrams, trigrams, and 4-grams.
- Weight all the scores equally with their logarithms.
- Sum the result and then take the exponent


### Brevity Penalty
This penalizes the scores if the length of the prediciton is not longer than the actual sentence's length by a factor

<img src="https://render.githubusercontent.com/render/math?math=\text{BP} =\exp\bigg(1 - \frac{\text{actual length}}{\text{prediction length}}\bigg)">

<!--- 
$\text{BP} =\exp\bigg(1 - \frac{\text{actual length}}{\text{prediction length}}\bigg)$
-->


## Interpretation
![Alt text](bleu_interp.png)
