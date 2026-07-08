import boto3
import os
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()

ses = boto3.client(
    "ses",
    aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name           = os.getenv("AWS_REGION")
)

SENDER_EMAIL   = os.getenv("SES_SENDER_EMAIL")
RECEIVER_EMAIL = os.getenv("SES_RECEIVER_EMAIL")

def send_failure_alert(ticket_id: int, error_detail: str,
                        ticket_text: str = "") -> bool:
    """
    Send an alert email ONLY when ticket processing fails.
    Not used for needs_review or Incident classification — those
    are surfaced via the dashboard, not email.
    """
    try:
        subject = f"🔴 Processing Failed — Ticket #{ticket_id}"

        body_html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">

            <div style="background: #dc2626; padding: 20px; border-radius: 8px 8px 0 0;">
                <h2 style="color: white; margin: 0;">
                    SmartTicket AI — Pipeline Failure
                </h2>
            </div>

            <div style="background: #f8fafc; padding: 24px;
                        border: 1px solid #e2e8f0; border-radius: 0 0 8px 8px;">

                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; color: #64748b; width: 140px;">
                            Ticket ID
                        </td>
                        <td style="padding: 8px; font-weight: bold;">
                            #{ticket_id}
                        </td>
                    </tr>
                    <tr style="background: white;">
                        <td style="padding: 8px; color: #64748b;">Status</td>
                        <td style="padding: 8px; font-weight: bold; color: #dc2626;">
                            FAILED
                        </td>
                    </tr>
                </table>

                <div style="margin-top: 16px; padding: 16px;
                            background: white; border-radius: 8px;
                            border-left: 4px solid #dc2626;">
                    <p style="color: #64748b; margin: 0 0 8px 0;
                               font-size: 12px; text-transform: uppercase;">
                        Error Detail
                    </p>
                    <p style="margin: 0; color: #1e293b; font-family: monospace;
                              font-size: 13px;">
                        {error_detail[:500]}
                    </p>
                </div>

                {"<div style='margin-top:16px;padding:16px;background:white;border-radius:8px;'>"
                 f"<p style='color:#64748b;font-size:12px;text-transform:uppercase;margin:0 0 8px 0;'>Ticket Text</p>"
                 f"<p style='margin:0;color:#1e293b;'>{ticket_text[:300]}</p></div>" if ticket_text else ""}

                <div style="margin-top: 20px; text-align: center;">
                    <p style="color: #94a3b8; font-size: 12px;">
                        SmartTicket AI — Automated Failure Alert
                    </p>
                </div>
            </div>

        </body>
        </html>
        """

        ses.send_email(
            Source      = SENDER_EMAIL,
            Destination = {"ToAddresses": [RECEIVER_EMAIL]},
            Message     = {
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body"   : {"Html": {"Data": body_html, "Charset": "UTF-8"}}
            }
        )
        logger.info(f"SES failure alert sent → Ticket #{ticket_id}")
        return True

    except Exception as e:
        logger.error(f"SES email failed: {str(e)}")
        return False