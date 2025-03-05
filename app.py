import typer

from src.commands.collect_and_store import collect_and_store_email
from src.commands.create_inbox_folders import create_inbox_folders
from src.commands.review_and_categorize import review_categorize_todays_email
from src.commands.train_topic_model import train_topic_model

app = typer.Typer(help="Email Organizer CLI")

# Attach each command to the `app`:
app.command(name="collect_and_store")(collect_and_store_email)
app.command(name="create_inbox_folders")(create_inbox_folders)
app.command(name="organize")(review_categorize_todays_email)
app.command(name="train_topic_model")(train_topic_model)

if __name__ == "__main__":
    app()
