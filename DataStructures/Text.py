import email


class Email:

    @staticmethod
    def from_content(text):
        return Email(text).body

    def __init__(self, text):
        """
        Create an email object from a string.

        Source: https://stackoverflow.com/questions/17874360/python-how-to-parse-the-body-from-a-raw-email-given-that-raw-email-does-not#32840516
        """
        msg = email.message_from_string(text)

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))

                if content_type == 'text/plain':
                    if 'attachment' not in content_disposition:
                        self.body = part.get_payload(decode=True)
                    else:
                        self.attachment = part.get_payload()

        else:
            self.body = msg.get_payload()

        self.raw = msg
