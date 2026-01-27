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
The training set is skew which brings the bias to the model that the model will tned to make more classfy on type'4'. We can also use stratified sampling which makes the traing set has equal number for each type. This can improve the model have better perofrmance on '0-3'. However, this model is not ideal for the reality. In real word, the model will have bad performance on type '4' since this product has over 60% type'4' rating.

We can use stratified sampling + class weight loss method to let the model learn more for type '0-3'.
This is the comparison with/without  'class weight loss'
confustion metrics comparision 
<p float="left">
  <img width="613" height="486" alt="image" src="https://github.com/user-attachments/assets/57462d34-4661-4c8a-84da-5b1733255f35" />
<img width="628" height="664" alt="image" src="https://github.com/user-attachments/assets/9b478d64-9ba4-46fd-98cb-d7a49b5f586c" />
</p>

accuracy comparision


 loss curve & leanring curve display
 


