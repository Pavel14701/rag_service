# RAG Service

Асинхронный сервис **RAG** (Retrieval-Augmented Generation) на Python 3.12: принимает документы и вопросы пользователей через RabbitMQ, индексирует документы в векторную базу и отвечает на вопросы на основе проиндексированного контента — с проверкой прав доступа на уровне векторного поиска.

## Возможности

- 📥 Индексация документов (Markdown, PDF, DOCX и др.) с чанкингом и эмбеддингами, OCR для сканов (`PDF_OCR_STRATEGY` / `PDF_OCR_LANGUAGES`)
- 🔎 Векторный поиск с фильтрацией по правам доступа (владелец / группа доступа)
- 🧩 Гибридный поиск: векторный + лексический BM25, слияние ранжированием RRF (`SEARCH_HYBRID=true`), graceful деградация в чисто векторный
- 💬 Генерация ответов через DeepSeek API строго по найденному контексту (температура — `LLM_TEMPERATURE`), circuit breaker на LLM API
- 🚀 Кэши: LLM-ответы и query-эмбеддинги в Redis (`REDIS_URL`) либо in-process TTL; недоступный Redis = cache miss, обработка не ломается
- 🖼 Поддержка мультимодального контента: таблицы сохраняются с HTML-структурой, элементы тегируются (`type`: table / image / title)
- 🏷 Маркер `embedding_model` в каждой точке Qdrant + предупреждение при старте о смене модели эмбеддингов (нужен переиндекс)
- 📊 Offline-оценка качества RAG: precision@k / recall@k / MRR / NDCG / faithfulness, A/B-сравнение конфигураций (`src/evaluation/`, `scripts/run_eval.py`)
- 🗑 Удаление документов (soft-delete + очистка векторов, опционально файла)
- 🔄 Полный реиндекс коллекции (админ-операция) с логированием миграции версии эмбеддингов
- 🔐 JWT-аутентификация каждого сообщения
- ♻️ Retry с exponential backoff + DLX для ошибочных сообщений
- 🏥 Health/metrics HTTP-сервер (liveness/readiness-пробы, Prometheus-метрики), трассировка OpenTelemetry
- ⚙️ Разделение ролей воркеров (`WORKER_QUEUES=query|background|all`) для раздельного масштабирования
- Полностью асинхронный воркер (asyncio + aio-pika), DI на dishka

## Стек

| Компонент | Технология |
|---|---|
| Runtime | Python 3.12+, asyncio, [uv](https://docs.astral.sh/uv/) |
| Брокер сообщений | RabbitMQ (aio-pika) |
| Векторная БД | Qdrant |
| Метаданные | PostgreSQL 15 + SQLAlchemy 2 (asyncio, asyncpg), миграции Alembic |
| Файловое хранилище | MinIO (S3 API) |
| Эмбеддинги | sentence-transformers, `intfloat/multilingual-e5-small` (384 dim) |
| LLM | DeepSeek Chat API (`deepseek-chat`) |
| Парсинг документов | unstructured (+ markdown / beautifulsoup4), OCR через tesseract |
| Кэши | Redis (опционально) / in-process TTLCache |
| DI | dishka |
| Конфигурация | pydantic-settings (`.env`) |
| Наблюдаемость | Prometheus-метрики, structlog, OpenTelemetry |
| Тесты | pytest + pytest-asyncio; мутационное тестирование mutmut (через WSL на Windows) |

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

- `src/entrypoints` — консьюмеры RabbitMQ (retry + DLX), health/metrics-сервер: валидация JWT, диспетчеризация в сервисы
- `src/application/services` — use-cases: `IndexerService`, `RetrieverService`, `DocumentManager`
- `src/application/interfaces` — порты (Protocol): `VectorStore`, `FileStorage`, `DocumentRepository`, `EmbeddingModel`, `LLMGenerator`, `TokenValidator`
- `src/domain` — сущности (`Document`, `Conversation`) и исключения
- `src/infrastructure` — адаптеры: Qdrant, MinIO, Postgres, Redis, sentence-transformers, DeepSeek (+ circuit breaker), JWT, парсеры
- `src/evaluation` — offline-оценка качества (метрики retrieval/faithfulness, golden-датасет, раннер A/B)
- `src/shared` — гибридный поиск (BM25+RRF), кэши (TTL/Redis), circuit breaker, метрики, трассировка, логирование, хеширование
- `src/container.py` — сборка графа зависимостей (dishka), `src/main.py` — запуск health-сервера и консьюмеров выбранной роли

## Быстрый старт

```bash
# 1. Инфраструктура (RabbitMQ, PostgreSQL, Qdrant, MinIO)
docker compose up -d

# 2. Конфигурация
cp .env.example .env
# укажите минимум: DEEPSEEK_API_KEY и JWT_SECRET

# 3. Зависимости (нужны Python 3.12+ и uv)
uv sync

# 4. Миграции БД
uv run alembic upgrade head

# 5. Запуск воркера
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
| Redis | 6379 | кэши LLM-ответов и query-эмбеддингов |
| Health / Metrics воркера | 8000 | `/healthz` (liveness), `/readyz` (readiness), `/metrics` (Prometheus) |

## Миграции БД

Схема (`documents`, `conversations`) управляется Alembic; миграции применяются автоматически одним из способов:

- вручную: `uv run alembic upgrade head` (URL берётся из `POSTGRES_DSN` — env или `.env`);
- в docker-compose: one-shot сервис `migrate` (`alembic upgrade head`) — оба воркера стартуют только после его успешного завершения.

Новая миграция после изменения ORM-моделей:

```bash
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head          # проверьте сгенерированный DDL перед применением
```

Тесты (`tests/test_migrations.py`) прогоняют миграцию в offline-режиме и сверяют DDL с ORM-метаданными — расхождение схемы и миграций ломает тесты.

## Роли воркеров

docker-compose поднимает два воркера: `WORKER_QUEUES=query` (real-time ответы) и `WORKER_QUEUES=background` (индексация/удаление/реиндекс). Значение `all` запускает все очереди в одном процессе (удобно для разработки).

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
| `WORKER_QUEUES` | `all` | роли воркера: `query` / `background` / `all` |
| `METRICS_PORT` | `8000` | порт health/metrics-сервера |
| `LLM_TEMPERATURE` | `0.1` | температура генерации |
| `LLM_CIRCUIT_FAILURE_THRESHOLD` / `LLM_CIRCUIT_RESET_TIMEOUT` | `5` / `60` | circuit breaker на LLM API |
| `REDIS_URL` | — | Redis для кэшей; не задан → in-process TTLCache |
| `REDIS_CACHE_PREFIX` | `rag` | префикс ключей Redis |
| `SEARCH_HYBRID` | `false` | гибридный поиск (BM25 + RRF поверх векторного) |
| `SEARCH_HYBRID_RRF_K` / `SEARCH_HYBRID_CANDIDATES` | `60` / `50` | параметры RRF и лимит лексических кандидатов |
| `PDF_OCR_STRATEGY` | `auto` | стратегия unstructured: `auto`/`hi_res`/`ocr_only`/`fast` |
| `PDF_OCR_LANGUAGES` | `eng` | языки tesseract через запятую (напр. `eng,rus`) |
| `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` | `query: ` / `passage: ` | префиксы e5-моделей |
| `QDRANT_HNSW_M` / `QDRANT_HNSW_EF_CONSTRUCT` / `QDRANT_HNSW_EF` | — | тюнинг HNSW (None = дефолты сервера) |
| `POSTGRES_READ_DSN` | — | опциональная read-реплика для запросов |
| `OTEL_ENDPOINT` | — | exporter OTLP (трассировка) |

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

**Индексация** (`ingest_queue`): проверяется, что запрашивающий — владелец документа → файл скачивается из MinIO во временный каталог → парсится подходящим парсером (`ParserFactory`: `.md`, `.pdf`, `.docx`, остальное — unstructured auto; для PDF — выбранная OCR-стратегия) → текст режется на чанки (~512 символов, по границам слов) → каждому чанку присваивается детерминированный UUID (`uuid5` от `doc_id:idx:chunk_idx` — повторная индексация обновляет точки, а не дублирует) → эмбеддинги (`passage: `-префикс) → upsert в Qdrant с payload `{text, doc_id, owner_id, access_group, page, header, type, table_html?, embedding_model}` → статус в Postgres: `pending → indexed` (или `failed`).

**Ответ на вопрос** (`query_queue`): вопрос эмбеддируется (с `query: `-префиксом; эмбеддинги кэшируются) → гибридный поиск по Qdrant: векторный top_k с фильтром доступа (`owner_id == user` ИЛИ `access_group` ∈ групп пользователя) + при `SEARCH_HYBRID=true` лексические кандидаты (full-text индекс по `text`) с BM25-ранжированием и RRF-слиянием → чанки собираются в контекст → LLM (DeepSeek, `LLM_TEMPERATURE`, кэш ответов, circuit breaker) генерирует ответ строго по контексту → вопрос, ответ и источники сохраняются в `conversations`.

**Удаление** (`delete_queue`): только владелец → удаление векторов документа из Qdrant → (опционально) удаление файла из MinIO → soft-delete в Postgres.

**Реиндекс** (`reindex_queue`): только админ (группа `admin` или `user_id` с префиксом `admin_`) → коллекция Qdrant пересоздаётся (с HNSW-настройками и full-text индексом при гибриде) → все активные документы индексируются заново с текущим маркером `embedding_model` (ошибка отдельного документа не прерывает процесс).

**Смена модели эмбеддингов**: при старте воркер сравнивает `EMBEDDING_MODEL` с маркером в существующей коллекции и при рассинхроне логирует предупреждение о необходимости реиндекса (безопаснее явной команды, чем автоматический drop).

**Оценка качества** (offline): golden-датасет с релевантными документами прогоняется через `RetrieverService`, агрегируются precision@k / recall@k / MRR / NDCG / faithfulness / refusal rate; `compare_reports()` даёт дельты двух конфигураций для A/B (например, гибрид on/off):

```bash
PYTHONPATH=src python scripts/run_eval.py --dataset eval/golden_dataset.json --json-out report.json
```

## Тесты

```bash
uv run pytest
```

Покрытие: сервисы (indexer / retriever / document manager), консьюмеры, конвертация фильтров Qdrant, гибридный поиск (BM25/RRF/full-text индекс), кэши (TTL + Redis), circuit breaker, offline-evaluation (метрики, датасет, раннер), мультимодальные payload'ы (тип элемента, таблицы), маркер версии эмбеддингов, JWT, парсеры (вкл. OCR-конфигурацию), ORM-маппинг, хеширование. Внешние зависимости заменены in-memory double'ами (`tests/conftest.py`) — БД, брокер и Redis для тестов не нужны.

> Тесты парсеров PDF/DOCX/unstructured автоматически помечаются `skip`, если в системе недоступен `libmagic`/`unstructured` (типично для чистой Windows; на Linux обычно установлен).

Мутационное тестирование (оценка качества самих тестов):

```bash
uv run mutmut run   # на Windows — только через WSL (native support: boxed/mutmut#397)
```

## Структура проекта

```
src/
├── main.py                    # health-сервер + консьюмеры выбранной роли
├── config.py                  # Settings (pydantic-settings)
├── container.py               # dishka-провайдеры
├── application/
│   ├── interfaces/            # порты (Protocol)
│   └── services/              # indexer, retriever, document_manager
├── domain/
│   ├── entities/              # Document, Conversation
│   └── exceptions.py
├── entrypoints/               # base_consumer (retry+DLX), 4 консьюмера, health-сервер
├── evaluation/                # offline-оценка качества: metrics, dataset, runner
├── infrastructure/
│   ├── embedding/             # sentence-transformers (+ префиксы e5, батчи)
│   ├── file_storage/          # MinIO
│   ├── llm/                   # DeepSeek HTTP-клиент (+ circuit breaker)
│   ├── parsing/               # markdown / pdf (OCR) / docx / unstructured + фабрика
│   ├── repositories/          # Postgres (read-реплика) + ORM-модели
│   ├── security/              # JWT
│   └── vector_store/          # Qdrant (+ гибрид, HNSW, scroll)
└── shared/                    # hybrid (BM25+RRF), caching (TTL/Redis), circuit_breaker,
                               # metrics (Prometheus), tracing (OTel), logging, hashing
alembic.ini                    # конфигурация Alembic
migrations/                    # env.py (async) + версии миграций
eval/golden_dataset.json       # golden-датасет для оценки качества
scripts/run_eval.py            # CLI запуска offline-evaluation
tests/                         # pytest, double'ы в conftest.py
```

## Известные ограничения

- **Ответ не возвращается продюсеру**: `QueryConsumer` пишет результат в stdout/conversation; предполагается reply-очередь.
- `get_user_groups()` в Postgres-репозитории — заглушка (возвращает пустой список); фильтр по группам заработает после интеграции с auth-сервисом (порт уже принимает группы).
- Для реального OCR (`ocr_only`/`hi_res` на сканах) требуются `tesseract` и `poppler` в образе воркера.
- `faithfulness` в evaluation — лексическая эвристика (n-граммное перекрытие с источниками), а не семантический судья; для строгой оценки подключите LLM-as-judge.
- mutmut на Windows запускается только через WSL.
- Golden-датасет (`eval/golden_dataset.json`) — образец с placeholder-идентификаторами; наполните реальными кейсами по вашим документам.
