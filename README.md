# final_year_project_2025

PDF Summariser

## Testing Script

- Make sure you are using the virtual environment

### Test Website

- Have 3 terminals open

#### Start Celery Worker

- Start Redis server on one terminal

```terminal
redis-server
```

- Run Celery on another terminal

```terminal
inv devworker
```

#### Start Web Server

- On a separate terminal run

```terminal
./start_web.sh
```

### Test Summary

- Run Async

```terminal
python3 -m tests.test_summary sync
```

- Run Multi-Threaded

```terminal
python3 -m tests.test_summary Async
