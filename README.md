# MAFIA QA (RAG) CONSULTANT

Цель проекта: консультант по регламенту игры "Спортивная мафия"

- однако архитектура **мгновенно адаптируется** под любой домен – нужн о заменить лишь документы и промпты в .env.
- В качестве LLM модели рекомендуем использовать  `HuggingFaceH4/zephyr-7b-beta`, а в качестве векторизатора –`sentence-transformers/distiluse-base-multilingual-cased-v1`. Это **open-source** модели высокого качества.

Проект состоит из двух сервисов:

- model-service – сервис для генерации текста моделью. Для удобства он в том же репозитории, но при необходимости может быть легко в отдельный сервис для запуска на отдельном сервере с gpu.
- qa-service принимает вопрос от пользователя, обращается к векторной базе данных для поиска релевантного контекста и отправляет вопрос в model-service. Именно он является точкой входа для пользователя.

Оба сервиса можно запустить как на одной машине с gpu, так и разделить – model-service на gpu-сервере, qa-service на локальной машине.

## Установка и запуск

### Step 1. Install the project

Для запуска на одном сервере:

```
poetry install --with model,call_qa
```

Для установки и последующего запуска только model-service используйте команду ниже на машине с gpu.

```
poetry install --with model
```

Для установки основного интерфейса обращений к ассистенту на любой машине.

```
poetry install --with call_qa

```

### Step 2. Заполнение .env и конфигов

1. Скопируйте файл .env.example, переименуйте его в .env и заполните. Описание параметров есть в .env.example.
2. Скопируйте файлы `mafia_qa/src/qa-service/api_config.yaml.example` и `mafia_qa/src/model-service/api_config.yaml.example` в соответсвующих родительских папках и для каждого сервиса укажите порт, на котором будет поднят сервис. Для model-service нужно также указать название Hugging Face LLM модели.

### Step 3. Заполнение базы

Если у вас нет FAISS базы с документами, заполните её `poetry run python src/qa-service/init_db.py`.

Скрипт ожидает, что в папке проекта будет находиться папка `documents`, в которой находится набор docx документов для загрузки в базу. Индекс FAISS базы будет сохранён в папке проекта в папке `faiss-index`.

### Step 4. Запуск сервисов

- для сервиса model-service ``poetry run python src/model-service/api.py``
- для сервиса qa `poetry run python src/qa-service/api.py`

## Эндпоинты

#### qa-service

**GET /healthcheck**

**Описание** : Проверяет доступность сервиса.

**Пример использования :**

```python
url = "http://158.160.71.89:5058/healthcheck"
response = requests.get(url)
response.text
```

**Ответ:**

 `'{"message":"Hey, hey! I've been alive for 0:02:59.871928 seconds now.\n"}'`

**POST /answer**

Получая вопрос пользователя, подтягивает контекст и вызывает модель для получения ответа.

**Параметры запроса**:

- `request`: Request - должен содержать поле 'query'.

**Пример использования**:

```python
QA_URL = "http://0.0.0.0:5055/answer"
query = "Можно ли спросить у судьи, кто выставлен на голосование?"
data = {"query": query}
response = requests.post(QA_URL, json=data, timeout=1000)
print(response.text['answer'])
```

**Ответ
**


**POST /load_document**

Загружает новые тексты в векторную базу данных и сохраняет обновленную базу.

 **Параметры запроса** :

* `request`: Request - должен содержать значение "documents" с списком текстов для загрузки.

 **Пример использования** :

```python
from docx import Document

LOAD_URL = "http://0.0.0.0:5055/load_document"
pathes = ['/documents/rules_addition.docx']
documents = []

for path in pathes:
    doc = Document(path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    documents.append(text)

data = {'documents': str(documents)}

response = requests.post(LOAD_URL, json=data, timeout=1000)
```

#### model-service

Поскольку все взаимодействия пользователя происходят через qa-service, пользователь может не беспокоить себя знанием о model-service.
