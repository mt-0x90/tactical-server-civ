import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from email.message import EmailMessage
from googleapiclient.discovery import build
import json
import time
import base64
from googleapiclient.errors import HttpError
import sys


class EmailHandler:
  def __init__(self, credentials_fpath):
    self.creds = None
    self.service = None
    self.scopes = ["https://www.googleapis.com/auth/gmail.readonly","https://www.googleapis.com/auth/gmail.send"]
    self.forward_to = "detectphishing.emails@gmail.com"
    self.tokens_fpath = "tokens.json"
    self.local_history = "local_history.json"
    self.msg_content = {}
    self.creds_fpath = credentials_fpath
    self.auth()

  def save_history(self,msgs):
    with open(self.local_history, 'w') as f:
      json.dump(msgs, f)
    f.close()

  def auth(self):
    if os.path.exists(self.tokens_fpath):
      self.creds = Credentials.from_authorized_user_file(self.tokens_fpath)
    if not self.creds or not self.creds.valid:
      if self.creds and self.creds.expired and self.creds.refresh_token:
        self.creds.refresh(Request())
      else:
        flow = InstalledAppFlow.from_client_secrets_file(self.creds_fpath, self.scopes)
        self.creds = flow.run_local_server(port=0)
      with open(self.tokens_fpath, "w") as f:
        f.write(self.creds.to_json())
      f.close()
    # Call the Gmail API
    self.service = build("gmail", "v1",credentials=self.creds)
    if not os.path.exists(self.local_history):
      self.save_history({"read":[]})

    


  def send_email(self, to_email:str, attachments: list=None):
    try:
      # Send the email as per Gmail API documentation
      message = EmailMessage()
      if attachments:
        for attachment in attachments:
          with open(attachment, 'rb') as content_file:
            content = content_file.read()
            message.add_attachment(content, maintype='application', subtype= (attachment.split('.')[1]), filename=attachment)
      message.set_content(f"[*] TACTICA AI [*]\n\n\nDear Operator,\n\n\nFind attached the report.")
      message['To'] = to_email
      message['From'] = "mthandazogegane@gmail.com"
      message['Subject'] = "Tactica AI Report"
      encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
      create_message = {"raw": encoded_message}
      send_message = (
        self.service.users().messages().send(userId="me", body=create_message).execute()
      )
      print(f"[*] Message sent to {self.forward_to}")
      print(f"[*] Message id: {send_message['id']}")
    except Exception as e:
      print(f"Exception: {e}")

# emailh = EmailHandler(sys.argv[1])
# emailh.send_email("mthandazondhlovu34@gmail.com", [sys.argv[2]])