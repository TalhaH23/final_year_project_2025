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
