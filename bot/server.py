"""SkillSieve Feishu Bot Server (Long Connection Mode).

Uses lark.ws.Client for WebSocket long connection — no public IP or ngrok needed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
)
from dotenv import load_dotenv

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.scanner import scan_skill_url, scan_skill_text
from bot.cards import build_scanning_card, build_result_card, build_error_card

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("skillsieve-bot")

APP_ID = os.environ["FEISHU_APP_ID"]
APP_SECRET = os.environ["FEISHU_APP_SECRET"]

# REST client for sending messages
api_client = lark.Client.builder() \
    .app_id(APP_ID) \
    .app_secret(APP_SECRET) \
    .log_level(lark.LogLevel.INFO) \
    .build()

# Dedup: avoid replying to the same message twice
_processed_messages: set = set()

# --------------------------------------------------------------------------
# Message handler
# --------------------------------------------------------------------------

GITHUB_URL_PATTERN = re.compile(
    r"https?://github\.com/[\w\-\.]+/[\w\-\.]+(?:/tree/[\w\-\.]+/[\w\-\.\/]+)?"
)


def handle_message(data):
    """Handle incoming message from Feishu via long connection."""
    event = data.event
    msg = event.message
    msg_id = msg.message_id
    chat_id = msg.chat_id

    # Dedup
    if msg_id in _processed_messages:
        return
    _processed_messages.add(msg_id)
    if len(_processed_messages) > 10000:
        _processed_messages.clear()

    # Parse message content — handle text, post, and rich text formats
    msg_type = msg.message_type
    try:
        content = json.loads(msg.content)
    except (json.JSONDecodeError, TypeError):
        content = {}

    logger.info(f"Received msg_type={msg_type} content={json.dumps(content, ensure_ascii=False)[:200]}")

    text = ""
    if msg_type == "text":
        text = content.get("text", "").strip()
    elif msg_type == "post":
        # Rich text: extract all text and links from post content
        parts = []
        post = content.get("content", [])
        for line in post:
            for elem in line:
                if elem.get("tag") == "text":
                    parts.append(elem.get("text", ""))
                elif elem.get("tag") == "a":
                    parts.append(elem.get("href", ""))
            parts.append(" ")
        text = " ".join(parts).strip()
    else:
        # Try to extract text from any format
        text = content.get("text", "")

    if not text:
        return

    # Remove @mention prefix (formats: @_user_1, @_all)
    text = re.sub(r"@_\w+\s*", "", text).strip()
    text = re.sub(r"@\S+\s*", "", text).strip()

    logger.info(f"Parsed text: {text[:150]}  chat={chat_id}")

    # Route
    url_match = GITHUB_URL_PATTERN.search(text)

    if url_match:
        url = url_match.group(0)
        _run_scan_async(msg_id, chat_id, url=url)
    elif any(kw in text.lower() for kw in ["扫描", "scan", "检测", "check", "analyze"]):
        skill_text = re.sub(r"(扫描|scan|检测|check|analyze)\s*", "", text, flags=re.IGNORECASE).strip()
        if skill_text:
            _run_scan_async(msg_id, chat_id, skill_text=skill_text)
        else:
            _send_text_reply(chat_id,
                             "请发送 GitHub URL 或技能内容。\n"
                             "示例: `扫描 https://github.com/user/skill-name`")
    elif text in ("/help", "帮助"):
        _send_text_reply(chat_id,
                         "**SkillSieve 安全扫描机器人**\n\n"
                         "用法:\n"
                         "1. 发送 GitHub URL → 自动扫描\n"
                         "   `https://github.com/user/skill-name`\n\n"
                         "2. 发送 `扫描 <内容>` → 分析技能文本\n\n"
                         "3. `/help` → 显示帮助\n\n"
                         "Layer 1 (静态分析) 免费，Layer 2/3 (LLM分析) 每天限10次")


def _run_scan_async(msg_id: str, chat_id: str, url: str = None, skill_text: str = None):
    """Run scan in background thread, update card as progress changes."""

    def _worker():
        # Step 1: Send "scanning" card
        card_msg_id = _send_card(chat_id, build_scanning_card(
            target=url or skill_text[:60],
            stage="Layer 1 静态分析中...",
        ))
        if not card_msg_id:
            logger.error("Failed to send initial card")
            return

        try:
            t0 = time.time()

            if url:
                result = scan_skill_url(url, progress_callback=lambda stage: _patch_card(
                    card_msg_id,
                    build_scanning_card(target=url, stage=stage)
                ))
            else:
                result = scan_skill_text(skill_text, progress_callback=lambda stage: _patch_card(
                    card_msg_id,
                    build_scanning_card(target=skill_text[:60], stage=stage)
                ))

            elapsed = time.time() - t0

            # Final result card
            _patch_card(card_msg_id, build_result_card(result, elapsed))
            logger.info(f"Scan done: {result.skill_name} → {result.verdict} ({elapsed:.1f}s)")

        except Exception as e:
            logger.exception(f"Scan failed: {e}")
            _patch_card(card_msg_id, build_error_card(str(e)))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


# --------------------------------------------------------------------------
# Feishu API helpers
# --------------------------------------------------------------------------

def _send_text_reply(chat_id: str, text: str):
    """Send a plain text reply."""
    req = CreateMessageRequest.builder() \
        .receive_id_type("chat_id") \
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("text")
            .content(json.dumps({"text": text}))
            .build()
        ).build()
    resp = api_client.im.v1.message.create(req)
    if not resp.success():
        logger.error(f"Send text failed: {resp.code} {resp.msg}")


def _send_card(chat_id: str, card: dict):
    """Send an Interactive Card and return the message_id."""
    req = CreateMessageRequest.builder() \
        .receive_id_type("chat_id") \
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("interactive")
            .content(json.dumps(card))
            .build()
        ).build()
    resp = api_client.im.v1.message.create(req)
    if resp.success():
        return resp.data.message_id
    else:
        logger.error(f"Send card failed: {resp.code} {resp.msg}")
        return None


def _patch_card(msg_id: str, card: dict):
    """Update an existing Interactive Card."""
    req = PatchMessageRequest.builder() \
        .message_id(msg_id) \
        .request_body(
            PatchMessageRequestBody.builder()
            .content(json.dumps(card))
            .build()
        ).build()
    resp = api_client.im.v1.message.patch(req)
    if not resp.success():
        logger.error(f"Patch card failed: {resp.code} {resp.msg}")


# --------------------------------------------------------------------------
# Long connection entry point
# --------------------------------------------------------------------------

def main():
    # Build event handler
    event_handler = lark.EventDispatcherHandler.builder("", "") \
        .register_p2_im_message_receive_v1(handle_message) \
        .build()

    # Create WebSocket long connection client
    ws_client = lark.ws.Client(
        app_id=APP_ID,
        app_secret=APP_SECRET,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    logger.info("SkillSieve Feishu Bot starting (long connection mode)...")
    logger.info("No public IP needed. Bot connects to Feishu via WebSocket.")
    logger.info("Press Ctrl+C to stop.")

    # start() blocks forever
    ws_client.start()


if __name__ == "__main__":
    main()
