# nlp-sentiment-pipeline
# BERT-Based Sentiment Analysis Pipeline with AWS SageMaker

## 🚀 Overview
This project demonstrates an end-to-end NLP pipeline for sentiment analysis on social media text using BERT. The model was fine-tuned on the Sentiment140 dataset using Google Colab, and then deployed as a scalable REST API via AWS SageMaker.

## 📌 Key Features
- Fine-tuned a BERT model on 10K+ tweets from the Sentiment140 dataset using Colab.
- Deployed the trained model as a RESTful endpoint using AWS SageMaker and Hugging Face SDK.
- Implemented regularization techniques (dropout, weight decay, early stopping) to prevent overfitting.
- Achieved 86%+ validation accuracy.

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
| Training Samples     | 10,000 tweets |
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

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
