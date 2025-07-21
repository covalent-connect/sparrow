from settings import aws_entrypoint
from utils.integrations.slack.send_message import send_slack_message
import traceback

if __name__ == "__main__":
    print("Running functions")
    try:
        aws_entrypoint.run_functions()
    except Exception as e:
        send_slack_message(f"Error running functions: {e}")
        send_slack_message(f"Traceback: {traceback.format_exc()}")
    

# python -m scripts.aws.spot.aws_entrypoint_sparrow