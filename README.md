# RAG Service

Асинхронный сервис **RAG** (Retrieval-Augmented Generation) на Python 3.12: принимает документы и вопросы пользователей через RabbitMQ, индексирует документы в векторную базу и отвечает на вопросы на основе проиндексированного контента — с проверкой прав доступа на уровне векторного поиска.

## Возможности

- 📥 Индексация документов (Markdown, PDF, DOCX и др.) с чанкингом и эмбеддингами
- 🔎 Векторный поиск с фильтрацией по правам доступа (владелец / группа доступа)
- 💬 Генерация ответов через DeepSeek API строго по найденному контексту
- 🗑 Удаление документов (soft-delete + очистка векторов, опционально файла)
- 🔄 Полный реиндекс коллекции (админ-операция)
- 🔐 JWT-аутентификация каждого сообщения
- Полностью асинхронный воркер (asyncio + aio-pika), DI на dishka

## Стек

| Компонент | Технология |
|---|---|
| Runtime | Python 3.12+, asyncio, [uv](https://docs.astral.sh/uv/) |
| Брокер сообщений | RabbitMQ (aio-pika) |
| Векторная БД | Qdrant |
| Метаданные | PostgreSQL 15 + SQLAlchemy 2 (asyncio, asyncpg) |
| Файловое хранилище | MinIO (S3 API) |
| Эмбеддинги | sentence-transformers, `intfloat/multilingual-e5-small` (384 dim) |
| LLM | DeepSeek Chat API (`deepseek-chat`) |
| Парсинг документов | unstructured (+ markdown / beautifulsoup4) |
| DI | dishka |
| Конфигурация | pydantic-settings (`.env`) |
| Тесты | pytest + pytest-asyncio |

## Архитектура

```
                  ┌──────────┐
  API / продюсер ─►  RabbitMQ │
                  └──────────┘
        ┌──────────────┬───────────────┬────────────────┐
        ▼              ▼               ▼                ▼
  ingest_queue   query_queue    delete_queue    reindex_queue
        │              │               │                │
        ▼              ▼               ▼                ▼
  IndexerService  RetrieverService   DocumentManager (delete / reindex)
        │              │               │                │
        ▼              ▼               ▼                ▼
     MinIO ──► Qdrant ◄──────────────┴────────────────┘
        │              │
        ▼              ▼
   PostgreSQL: documents, conversations
```

Слои проекта:

- `src/entrypoints` — консьюмеры RabbitMQ: валидация JWT, диспетчеризация в сервисы
- `src/application/services` — use-cases: `IndexerService`, `RetrieverService`, `DocumentManager`
- `src/application/interfaces` — порты (Protocol): `VectorStore`, `FileStorage`, `DocumentRepository`, `EmbeddingModel`, `LLMGenerator`, `TokenValidator`
- `src/domain` — сущности (`Document`, `Conversation`) и исключения
- `src/infrastructure` — адаптеры: Qdrant, MinIO, Postgres, sentence-transformers, DeepSeek, JWT, парсеры
- `src/container.py` — сборка графа зависимостей (dishka), `src/main.py` — запуск всех консьюмеров

## Быстрый старт

```bash
# 1. Инфраструктура (RabbitMQ, PostgreSQL, Qdrant, MinIO)
docker compose up -d

# 2. Конфигурация
cp .env.example .env
# укажите минимум: DEEPSEEK_API_KEY и JWT_SECRET

# 3. Зависимости (нужны Python 3.12+ и uv)
uv sync

# 4. Запуск воркера
uv run python src/main.py
```

Воркер подключится ко всем очередям и начнёт обрабатывать сообщения. Бакет MinIO (`documents`) создаётся автоматически при старте.

### Порты инфраструктуры

| Сервис | Порт | Примечание |
|---|---|---|
| RabbitMQ AMQP | 5672 | guest / guest |
| RabbitMQ Management UI | 15672 | http://localhost:15672 |
| PostgreSQL | 5432 | db / user / pass: `rag` / `rag` / `rag` |
| Qdrant HTTP / gRPC | 6333 / 6334 | дашборд: http://localhost:6333/dashboard |
| MinIO API / Console | 9000 / 9001 | minioadmin / minioadmin |

## Конфигурация

Переменные читаются из `.env` и окружения (полный список — в `.env.example`):

| Переменная | По умолчанию | Описание |
|---|---|---|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | подключение к RabbitMQ |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | векторная БД |
| `POSTGRES_DSN` | `postgresql+asyncpg://rag:rag@localhost:5432/rag` | метаданные |
| `MINIO_ENDPOINT` | `localhost:9000` | S3-хранилище |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` | ключи MinIO |
| `MINIO_BUCKET` | `documents` | бакет с исходными файлами |
| `DEEPSEEK_API_KEY` | **обязательно** | ключ DeepSeek API |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | базовый URL LLM API |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | модель эмбеддингов |
| `COLLECTION_NAME` | `documents` | коллекция Qdrant |
| `JWT_SECRET` | **обязательно** | секрет HS256 для токенов |
| `LOG_LEVEL` | `INFO` | уровень логирования |

## Очереди и формат сообщений

Все сообщения — JSON. Каждый запрос содержит `token` — JWT (HS256, подписан `JWT_SECRET`), в claim `sub` — ID пользователя. Сообщения с невалидным токеном или недостатком прав отклоняются.

| Очередь | Назначение | Payload |
|---|---|---|
| `ingest_queue` | проиндексировать документ | `token`, `doc_id` (UUID) |
| `query_queue` | ответить на вопрос (RAG) | `token`, `query`, `query_id?`, `conversation_id?` |
| `delete_queue` | удалить документ | `token`, `doc_id`, `remove_file?` (по умолчанию `false`) |
| `reindex_queue` | пересобрать коллекцию | `token`, `vector_dimension?` |

Пример публикации:

```python
import json
import uuid

import aio_pika


async def main() -> None:
    conn = await aio_pika.connect_robust("amqp://guest:guest@localhost:5672/")
    async with conn:
        ch = await conn.channel()
        await ch.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps({
                    "token": make_jwt(user_id="user-1"),  # ваш HS256-токен с claim `sub`
                    "doc_id": str(uuid.uuid4()),
                }).encode()
            ),
            routing_key="ingest_queue",
        )
```

## Как это работает

**Индексация** (`ingest_queue`): проверяется, что запрашивающий — владелец документа → файл скачивается из MinIO во временный каталог → парсится подходящим парсером (`ParserFactory`: `.md`, `.pdf`, `.docx`, остальное — unstructured auto) → текст режется на чанки (~512 символов, по границам слов) → каждому чанку присваивается детерминированный UUID (`uuid5` от `doc_id:idx:chunk_idx` — повторная индексация обновляет точки, а не дублирует) → эмбеддинги → upsert в Qdrant с payload `{text, doc_id, owner_id, access_group, page, header}` → статус в Postgres: `pending → indexed` (или `failed`).

**Ответ на вопрос** (`query_queue`): запрос эмбеддится → векторный поиск по Qdrant с фильтром доступа (`owner_id == user` ИЛИ `access_group` ∈ групп пользователя, top_k = 5) → чанки собираются в контекст → LLM (DeepSeek, temperature = 0.1) генерирует ответ строго по контексту → вопрос, ответ и источники сохраняются в `conversations`.

**Удаление** (`delete_queue`): только владелец → удаление векторов документа из Qdrant → (опционально) удаление файла из MinIO → soft-delete в Postgres.

**Реиндекс** (`reindex_queue`): только админ (группа `admin` или `user_id` с префиксом `admin_`) → коллекция Qdrant пересоздаётся → все активные документы индексируются заново (ошибка отдельного документа не прерывает процесс).

## Тесты

```bash
uv run pytest
```

Покрытие: сервисы (indexer / retriever / document manager), консьюмеры, конвертация фильтров Qdrant, JWT, парсеры, ORM-маппинг, хеширование. Внешние зависимости заменены in-memory double'ами (`tests/conftest.py`) — БД и брокер для тестов не нужны.

> 4 теста парсеров PDF/DOCX/unstructured автоматически помечаются `skip`, если в системе недоступен `libmagic` (типично для чистой Windows; на Linux обычно установлен).

## Структура проекта

```
src/
├── main.py                    # запуск всех консьюмеров
├── config.py                  # Settings (pydantic-settings)
├── container.py               # dishka-провайдеры
├── application/
│   ├── interfaces/            # порты (Protocol)
│   └── services/              # indexer, retriever, document_manager
├── domain/
│   ├── entities/              # Document, Conversation
│   └── exceptions.py
├── entrypoints/               # base_consumer + 4 консьюмера
├── infrastructure/
│   ├── embedding/             # sentence-transformers
│   ├── file_storage/          # MinIO
│   ├── llm/                   # DeepSeek HTTP-клиент
│   ├── parsing/               # markdown / pdf / docx / unstructured + фабрика
│   ├── repositories/          # Postgres
│   ├── security/              # JWT
│   └── vector_store/          # Qdrant
└── shared/                    # hashing, logging
tests/                         # pytest, double'ы в conftest.py
```

## Известные ограничения

- **Нет миграций БД.** ORM-модели описаны в `src/infrastructure/repositories/postgres_repo.py` (`documents`, `conversations`), но DDL не применяется автоматически — создайте таблицы (`Base.metadata.create_all(engine)`) или подключите Alembic.
- **Ответ не возвращается продюсеру**: `QueryConsumer` пишет результат в stdout; предполагается reply-очередь.
- `get_user_groups()` — заглушка (возвращает пустой список); фильтр по группам заработает после её реализации.
- Ошибочные сообщения ack-аются и логируются; DLX / retry-политика не настроены.
- Для e5-моделей рекомендуется префиксовать тексты (`query: …` / `passage: …`) — сейчас этого не делается.
- Нет Dockerfile для самого воркера.
