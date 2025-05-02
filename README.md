# final_year_project_2025

PDF Summariser

## Testing Script

- Make sure your are using the virtual environment

### Test Website

- Have 3 terminals open

#### Start Celery Worker

- Start Redis server on one terminal

```terminal
redis-server
```

- Run Celery worker on another terminal

```terminal
inv devworker
```

#### Start Web Server

- On a separate terminal:

- Remove old database if it exists

```terminal
rm test.db
```

- Initialise database

```terminal
python3 -m web.db.init_db
```

- Run Website

```terminal
inv dev
```

### Test Summary

- Run Async

```terminal
python3 -m tests.test_summary sync
```

- Run Multi-Threaded

```terminal
python3 -m tests.test_summary Async
