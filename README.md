# IMDB Sentiment Classifier

Классификатор тональности отзывов на фильмы на основе DistilBERT. Обучен на датасете IMDB, достигает точности **90.47%** на тестовой выборке.

## О проекте

Модель определяет, является ли отзыв положительным или отрицательным. В основе — предобученный DistilBERT (лёгкая версия BERT от Hugging Face), к которому добавлена классификационная голова. Обучение выполнялось на GPU (Google Colab, T4).

## Стек

- Python 3.10+
- PyTorch
- Hugging Face Transformers (DistilBERT)
- Hugging Face Datasets
- Google Colab (GPU T4)

## Структура проекта
.  
├── train.py # обучение модели  
├── predict.py # предсказание на новом тексте  
├── review_classifier.py # архитектура модели  
├── requirements.txt  
├── README.md  
└── .gitignore  

text

После обучения в папке `model/` появятся:

- `config.json`, `vocab.txt`, `tokenizer.json` — токенизатор
- `model.safetensors` — backbone DistilBERT
- `classifier.pth` — веса классификационной головы

## Установка

```bash
pip install -r requirements.txt
