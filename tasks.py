import os
from invoke import task

# @task
# def dev(ctx):
#     """Run the Flask development server."""
#     ctx.run(
#         "python3 -m web.main",
#         pty=os.name != "nt",
#         env={"FLASK_ENV": "development"}
#     )


@task
def dev(ctx):
    """Run the FastAPI development server."""
    ctx.run(
        "uvicorn web.main_fastapi:app --reload",
        pty=os.name != "nt",
        env={"ENV": "development"}
    )

@task
def devworker(ctx):
    ctx.run(
        "watchmedo auto-restart --directory=./app --pattern=*.py --recursive -- celery -A app.celery.worker worker --concurrency=1 --loglevel=INFO --pool=solo",
        pty=os.name != "nt",
        env={"APP_ENV": "development"},
    )