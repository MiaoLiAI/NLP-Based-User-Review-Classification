#### ***BERT-based User Review classification Model***
Text analysis can be performed using various language models such as BERT, GPT, and LLaMA. Among these, BERT is a bidirectional transformer model, which allows it to capture contextual information from both the left and right sides of a text. This bidirectional understanding enables a deeper and more accurate interpretation of language meaning.
BERT is particularly effective for short to medium-length text and excels at classification tasks. Since the goal of this project is to classify user reviews into five discrete rating categories (1–5 stars), BERT is a well-suited choice. Its strong contextual representation and proven performance in sentiment and text classification tasks make it ideal for predicting user review ratings accurately.


First Step：
Download dataset：https://d396qusza40orc.cloudfront.net/phoenixassets/amazon_baby.csv
<img width="835" height="390" alt="image" src="https://github.com/user-attachments/assets/10f4f44b-5aeb-454c-8838-e9083a6227bf" />
Second Step：
Text processing：this include lower capitall text, remove punctuations, null, etc;
Third Step:
Applied BERT model to tokenize the text (BertTokenizer.from_pretrained("bert-base-uncased"))
<img width="2284" height="770" alt="image" src="https://github.com/user-attachments/assets/bd6d8095-63ae-46bf-83f0-fb68a8563e4a" />
Fourth Setp:
Convert all the text tokens into ids, and 
# cleaned_data['token_id'] = cleaned_data['review_token'].apply(lambda x: tokenizer.convert_tokens_to_ids(x))
# cleaned_data['cls_token_sep_id'] = cleaned_data['review_token'].apply(lambda x: tokenizer.build_inputs_with_special_tokens(x))

