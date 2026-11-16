"""
Email notification service for MeetMind action items.

Sends one email per unique owner found in a session's action items.
Each email lists only that person's tasks — not the whole meeting's.

Why Resend instead of SendGrid or SMTP:
  - Resend has a generous free tier (3,000 emails/month)
  - Single API call, no SMTP config, no TLS setup
  - Official Python SDK: pip install resend
  - SendGrid works identically — just swap _send_via_resend for _send_via_sendgrid

Email provider is selected by RESEND_API_KEY in .env.
If neither key is set, the service logs what it would send (dev mode).

Usage:
  from app.services.email_service import send_action_item_emails
  results = send_action_item_emails(session, owner_emails)

  owner_emails maps speaker names to email addresses:
  {"Alice": "alice@company.com", "Bob": "bob@company.com"}
  Any owner not in owner_emails is skipped (not an error).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class EmailResult:
    owner: str
    email: str
    sent: bool
    error: str | None = None


def _build_html(
    owner: str,
    session_title: str,
    session_date: str,
    action_items: list[dict],
) -> str:
    """
    Builds a clean, readable HTML email body.
    Inline styles only — no external CSS, works in all email clients.
    """
    rows = ""
    for item in action_items:
        due = item.get("due") or "—"
        task = item.get("task", "")
        rows += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #f0f0f0;
                     font-size:14px;color:#1a1a1a;line-height:1.5;">
            {task}
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #f0f0f0;
                     font-size:14px;color:#6b7280;white-space:nowrap;">
            {due}
          </td>
        </tr>"""

    return f"""
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f9fafb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f9fafb;padding:40px 20px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0"
             style="background:#ffffff;border-radius:12px;overflow:hidden;
                    box-shadow:0 1px 3px rgba(0,0,0,0.08);">

        <!-- header -->
        <tr>
          <td style="background:#0f172a;padding:28px 32px;">
            <p style="margin:0;font-size:11px;color:#94a3b8;
                      text-transform:uppercase;letter-spacing:0.08em;">MeetMind</p>
            <h1 style="margin:8px 0 0;font-size:22px;font-weight:600;color:#f8fafc;">
              Your action items
            </h1>
          </td>
        </tr>

        <!-- greeting -->
        <tr>
          <td style="padding:28px 32px 0;">
            <p style="margin:0;font-size:15px;color:#374151;line-height:1.6;">
              Hi <strong>{owner}</strong>, here are the tasks assigned to you
              from <strong>{session_title}</strong>
              <span style="color:#6b7280;font-size:13px;">({session_date})</span>.
            </p>
          </td>
        </tr>

        <!-- action items table -->
        <tr>
          <td style="padding:20px 32px;">
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">
              <thead>
                <tr style="background:#f8fafc;">
                  <th style="padding:10px 12px;text-align:left;font-size:11px;
                             color:#6b7280;text-transform:uppercase;
                             letter-spacing:0.06em;font-weight:500;">Task</th>
                  <th style="padding:10px 12px;text-align:left;font-size:11px;
                             color:#6b7280;text-transform:uppercase;
                             letter-spacing:0.06em;font-weight:500;white-space:nowrap;">Due</th>
                </tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
          </td>
        </tr>

        <!-- footer -->
        <tr>
          <td style="padding:20px 32px 32px;">
            <p style="margin:0;font-size:12px;color:#9ca3af;line-height:1.6;">
              This email was sent by MeetMind after your meeting was processed.
              Tasks were extracted automatically from the transcript.
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def _build_plain(owner: str, session_title: str, action_items: list[dict]) -> str:
    lines = [f"Hi {owner},", f"", f"Your action items from: {session_title}", ""]
    for item in action_items:
        due = f" (due: {item['due']})" if item.get("due") else ""
        lines.append(f"  - {item['task']}{due}")
    lines += ["", "— MeetMind"]
    return "\n".join(lines)


def _send_via_resend(
    to: str,
    subject: str,
    html: str,
    plain: str,
    from_address: str,
    api_key: str,
) -> None:
    """Send via Resend API. Raises on failure."""
    try:
        import resend
    except ImportError as exc:
        raise RuntimeError(
            "resend package not installed. Run: pip install resend"
        ) from exc

    resend.api_key = api_key
    resend.Emails.send({
        "from": from_address,
        "to": [to],
        "subject": subject,
        "html": html,
        "text": plain,
    })


def _log_dev_mode(to: str, subject: str, plain: str) -> None:
    """Dev mode — log instead of sending. No API key needed."""
    logger.info(
        "DEV MODE — email not sent (set RESEND_API_KEY to enable)\n"
        f"  To:      {to}\n"
        f"  Subject: {subject}\n"
        f"  Body:\n{plain}"
    )


def send_action_item_emails(
    session_id: str,
    session_title: str,
    session_date: str,
    action_items_json: list[dict],
    owner_emails: dict[str, str],   # {"Alice": "alice@example.com", ...}
) -> list[EmailResult]:
    """
    Groups action items by owner and sends one email per owner.

    Args:
        session_id:        used for logging only
        session_title:     shown in email subject and body
        session_date:      human-readable date string shown in body
        action_items_json: list of action item dicts from professional_output_json
        owner_emails:      mapping from speaker name → email address

    Returns:
        list[EmailResult] — one per owner who has an email address
    """
    import os
    resend_key = os.getenv("RESEND_API_KEY", "")
    from_address = os.getenv("EMAIL_FROM", "MeetMind <notifications@meetmind.app>")
    dev_mode = not resend_key

    # group by owner
    by_owner: dict[str, list[dict]] = {}
    for item in action_items_json:
        owner = item.get("owner", "Unknown")
        by_owner.setdefault(owner, []).append(item)

    results: list[EmailResult] = []

    for owner, items in by_owner.items():
        email = owner_emails.get(owner)
        if not email:
            logger.debug("No email address for owner %r — skipping", owner)
            continue

        subject = f'Your action items from "{session_title}"'
        html = _build_html(owner, session_title, session_date, items)
        plain = _build_plain(owner, session_title, items)

        if dev_mode:
            _log_dev_mode(email, subject, plain)
            results.append(EmailResult(owner=owner, email=email, sent=True))
            continue

        try:
            _send_via_resend(email, subject, html, plain, from_address, resend_key)
            logger.info(
                "Action item email sent",
                extra={"session_id": session_id, "owner": owner, "to": email},
            )
            results.append(EmailResult(owner=owner, email=email, sent=True))
        except Exception as exc:
            logger.error(
                "Failed to send action item email",
                extra={"session_id": session_id, "owner": owner, "error": str(exc)},
            )
            results.append(EmailResult(owner=owner, email=email, sent=False, error=str(exc)))

    return results
