# nlp-sentiment-pipeline
# BERT-Based Sentiment Analysis Pipeline with AWS SageMaker

## 🚀 Overview
This project demonstrates an end-to-end NLP pipeline for sentiment analysis on social media text using BERT. The model was fine-tuned on the Sentiment140 dataset using Google Colab, and then deployed as a scalable REST API via AWS SageMaker.

## 🤖 About the BERT Model
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model introduced by Google. It is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. The core of BERT includes:
- Multiple transformer encoder layers
- Multi-head self-attention mechanisms
- Positional embeddings
- A classification head for downstream tasks like sentiment analysis

In this project, we used the pre-trained `bert-base-uncased` model with a classification head for binary sentiment classification (positive vs. negative).

### 📊 BERT Architecture Diagram

![BERT Architecture](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*-AHw6mjJc4rVbBtx3U0U8g.png)

> *Source: Devlin et al., 2018*

### 📚 Citation

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).  
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). *arXiv:1810.04805*

## 📌 Key Features
- Fine-tuned a BERT model on 10K+ tweets from the Sentiment140 dataset using Colab.
- Deployed the trained model as a RESTful endpoint using AWS SageMaker and Hugging Face SDK.
- Implemented regularization techniques (dropout, weight decay, early stopping) to prevent overfitting.
- Achieved 86%+ validation accuracy.

## ⚙️ Training at Scale: Handling 1.6M Tweets
The Sentiment140 dataset contains 1.6 million labeled tweets. Due to memory and compute limitations on Colab and SageMaker Free Tier:
- We used `pandas.read_csv(..., chunksize=1000)` to stream data in mini-batches from disk.
- Each chunk was tokenized, converted into PyTorch tensors, and used to train the model incrementally.
- Between chunks, we cleared GPU memory using `gc.collect()` and `torch.cuda.empty_cache()` to avoid OOM errors.

This strategy allowed us to perform scalable training in constrained environments.

## 🔥 Overcoming Limitations and Overfitting
### Challenges:
- **Free Tier Constraints**: No GPU and limited RAM on SageMaker Free Tier.
- **Overfitting**: Small batch sizes and few samples per chunk initially led to high training accuracy but poor generalization.

### Solutions:
- Added `dropout=0.3` and `attention_probs_dropout_prob=0.3` in BERT config.
- Used `weight_decay=0.01` with `AdamW` optimizer.
- Introduced early stopping to halt training if validation accuracy plateaued.
- Stratified splitting of a separate validation set to ensure balanced label representation.

## 🔬 Loss Function and Optimization
- **Loss Function**: Binary Cross Entropy via `CrossEntropyLoss`, applied to logits output from the classification head.
- **Optimizer**: AdamW (`torch.optim.AdamW`) with learning rate = 2e-5 and weight decay = 0.01.
- **Gradient Descent**: Standard mini-batch backpropagation, gradients computed via `.backward()` and updated via `optimizer.step()`.

## 🛠️ Technologies Used
- Python 3
- PyTorch & Hugging Face Transformers
- Google Colab (Training)
- AWS SageMaker (Deployment)
- Pandas, Scikit-learn

## 🗂️ Project Structure
```
bert-sentiment-analysis/
├── bert_sentiment/               # Trained model files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── bert_sentiment.zip           # Zipped model for upload
├── train_model.ipynb            # Colab notebook for training
├── deploy_model.ipynb           # SageMaker Studio notebook for deployment
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
```

## 📈 Results
| Metric               | Value        |
|----------------------|--------------|
| Validation Accuracy  | 86%          |
| Training Samples     | 10,000+ tweets |
| Inference Latency    | ~200ms       |

## 🌐 API Usage Example
After deployment via SageMaker:
```bash
POST https://<your-sagemaker-endpoint>
{
  "inputs": "I absolutely love this product!"
}
```
Response:
```json
{
  "label": "POSITIVE",
  "score": 0.982
}
```

## 👨‍💻 Author
Pavia Bera  
PhD in AI & Machine Learning  
University of South Florida



## 📚 References
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). arXiv:1810.04805.


## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

