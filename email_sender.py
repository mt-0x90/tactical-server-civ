from google_script import Create_Service
import os
import base64
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import mimetypes
 
CLIENT_SECRET_FILE = sys.argv[2]
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/']
 
service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
 
file_attachments = [sys.argv[1]]
 
emailMsg = 'Three files attached'
 
# create email message
mimeMessage = MIMEMultipart()
mimeMessage['to'] = 'mthandazo.ndhlovu@.tii.ae'
mimeMessage['subject'] = 'Tactica Operator'
mimeMessage.attach(MIMEText(emailMsg, 'plain'))
 
# Attach files
for attachment in file_attachments:
    content_type, encoding = mimetypes.guess_type(attachment)
    main_type, sub_type = content_type.split('/', 1)
    file_name = os.path.basename(attachment)
 
    f = open(attachment, 'rb')
 
    myFile = MIMEBase(main_type, sub_type)
    myFile.set_payload(f.read())
    myFile.add_header('Content-Disposition', 'attachment', filename=file_name)
    encoders.encode_base64(myFile)
 
    f.close()
 
    mimeMessage.attach(myFile)
 
raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()
 
message = service.users().messages().send(
    userId='me',
    body={'raw': raw_string}).execute()
 
print(message)