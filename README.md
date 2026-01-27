#### ***BERT-based User Review classification Model***
Text analysis can be performed using various language models such as BERT, GPT, and LLaMA. Among these, BERT is a bidirectional transformer model, which allows it to capture contextual information from both the left and right sides of a text. This bidirectional understanding enables a deeper and more accurate interpretation of language meaning.
BERT is particularly effective for short to medium-length text and excels at classification tasks. Since the goal of this project is to classify user reviews into five discrete rating categories (1–5 stars), BERT is a well-suited choice. Its strong contextual representation and proven performance in sentiment and text classification tasks make it ideal for predicting user review ratings accurately.

**Part One**
Download Dataset
source：https://d396qusza40orc.cloudfront.net/phoenixassets/amazon_baby.csv
<img width="835" height="390" alt="image" src="https://github.com/user-attachments/assets/10f4f44b-5aeb-454c-8838-e9083a6227bf" />
Text Preprocessing
Convert text to lowercase
Remove punctuation
Handle null/missing values
BERT Tokenization:
Applied BERT model to tokenize the text (BertTokenizer.from_pretrained("bert-base-uncased"))
<img width="2284" height="770" alt="image" src="https://github.com/user-attachments/assets/bd6d8095-63ae-46bf-83f0-fb68a8563e4a" />
Convert tokens to IDs:

cleaned_data['token_id'] = cleaned_data['review_token'].apply(lambda x: tokenizer.convert_tokens_to_ids(x))

Add [CLS] and [SEP] tokens:
cleaned_data['cls_token_sep_id'] = cleaned_data['review_token'].apply(lambda x: tokenizer.build_inputs_with_special_tokens(x))

<img width="497" height="128" alt="image" src="https://github.com/user-attachments/assets/d43de519-e388-4d87-8081-476c8a0b2e03" />

Add Attention Mask
Attention masks indicate which tokens the model should focus on and which to ignore.

<img width="318" height="127" alt="image" src="https://github.com/user-attachments/assets/1329f19e-442d-4c70-9ff9-4a08ddda8b29" />

**Train Model & Classification**

Used 10,000 records for initial training due to dataset size.
Confusion matrix shows the model predicts class '4' well, but performs poorly on other classes.

Training data is skewed: over 50% of reviews are class '4', so the model naturally tends to predict '4'.

<img width="228" height="191" alt="image" src="https://github.com/user-attachments/assets/6a12d1cc-7fac-48de-b5b2-2bbe4756f7b8" />

**Training Set Bias**

The training set is skewed, with over 50% of reviews in class '4', which causes the model to bias toward predicting '4'.

Stratified sampling can balance the training set to improve performance on classes '0–3', but this may not reflect real-world distribution (type '4' is still >60% in reality).

Solution:

Use stratified sampling + class-weighted loss to let the model learn more from minority classes (type '0–3') while keeping majority class distribution realistic.

Comparison of confusion matrices with and without class-weighted loss:

<p float="left"> <img width="300" height="328" src="https://github.com/user-attachments/assets/4be727f5-8c0a-45aa-9c2c-5bf06815d4c3" /> <img width="300" height="328" src="https://github.com/user-attachments/assets/9b478d64-9ba4-46fd-98cb-d7a49b5f586c" /> </p>

Softmax Probability Adjustment for Imbalanced Classes

In an imbalanced classification scenario, the model tends to predict the majority class most of the time. To improve minority class recall without changing the training distribution, we can adjust the probabilities after the softmax layer during inference:

Softmax: Converts the model's logits into probabilities for each class.

Adjustment: Multiply minority class probabilities by a factor >1 (e.g., 1.2) and then renormalize so the probabilities sum to 1.

Argmax Prediction: The class with the highest adjusted probability becomes the predicted label.

Effect:

Minority class probabilities are effectively “boosted,” making the model more likely to predict them.

Majority class probabilities slightly decrease, but the overall distribution remains interpretable.

This is equivalent to lowering the “implicit threshold” for minority class prediction, improving recall/F1 without retraining or oversampling.
 


