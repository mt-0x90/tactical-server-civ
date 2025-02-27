import base64
import mimetypes
import os
from email.message import EmailMessage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import sys
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def gmail_create_draft_with_attachment():
  """Create and insert a draft email with attachment.
   Print the returned draft's message and id.
  Returns: Draft object, including draft id and message meta data.

  Load pre-authorized user credentials from the environment.
  TODO(developer) - See https://developers.google.com/identity
  for guides on implementing OAuth2 for the application.
  """
  creds, _ = google.auth.default()

  try:
    # create gmail api client
    service = build("gmail", "v1", credentials=creds)
    mime_message = EmailMessage()

    # headers
    mime_message["To"] = "mthandazo.ndhlovu@tii.ae"
    mime_message["From"] = "mthandazogegane@gmail.com"
    mime_message["Subject"] = "sample with attachment"

    # text
    mime_message.set_content(
        "Hi, this is automated mail with attachment.Please do not reply."
    )

    # attachment
    attachment_filename = sys.argv[1]
    # guessing the MIME type
    type_subtype, _ = mimetypes.guess_type(attachment_filename)
    maintype, subtype = type_subtype.split("/")

    with open(attachment_filename, "rb") as fp:
      attachment_data = fp.read()
    mime_message.add_attachment(attachment_data, maintype, subtype)

    encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()

    create_draft_request_body = {"message": {"raw": encoded_message}}
    # pylint: disable=E1101
    # create_message = {"raw": encoded_message}
    # pylint: disable=E1101
    send_message = (
        service.users()
        .messages()
        .send(userId="me", body=create_draft_request_body)
        .execute()
    )
    print(f'Draft id: {send_message["id"]}\nDraft message: {send_message["message"]}')
  except HttpError as error:
    print(f"An error occurred: {error}")
    draft = None
  return draft


def build_file_part(file):
  """Creates a MIME part for a file.

  Args:
    file: The path to the file to be attached.

  Returns:
    A MIME part that can be attached to a message.
  """
  content_type, encoding = mimetypes.guess_type(file)

  if content_type is None or encoding is not None:
    content_type = "application/octet-stream"
  main_type, sub_type = content_type.split("/", 1)
  if main_type == "text":
    with open(file, "rb"):
      msg = MIMEText("r", _subtype=sub_type)
  elif main_type == "image":
    with open(file, "rb"):
      msg = MIMEImage("r", _subtype=sub_type)
  elif main_type == "audio":
    with open(file, "rb"):
      msg = MIMEAudio("r", _subtype=sub_type)
  else:
    with open(file, "rb"):
      msg = MIMEBase(main_type, sub_type)
      msg.set_payload(file.read())
  filename = os.path.basename(file)
  msg.add_header("Content-Disposition", "attachment", filename=filename)
  return msg


if __name__ == "__main__":
  gmail_create_draft_with_attachment()