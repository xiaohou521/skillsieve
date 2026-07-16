"""Feishu Interactive Card templates for SkillSieve Bot."""
from __future__ import annotations

from typing import Optional


def build_scanning_card(target: str, stage: str) -> dict:
    """Build a 'scanning in progress' card."""
    return {
        "header": {
            "template": "blue",
            "title": {
                "tag": "plain_text",
                "content": "🔍 SkillSieve 安全扫描"
            }
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**目标**\n{_escape(target[:80])}"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**状态**\n⏳ {stage}"
                        }
                    }
                ]
            },
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "SkillSieve - AI Agent 技能安全检测框架"
                    }
                ]
            }
        ]
    }


def build_result_card(result, elapsed: float) -> dict:
    """Build scan result card.

    Parameters
    ----------
    result : ScanResult
        The scan result object.
    elapsed : float
        Total time in seconds.
    """
    verdict = result.verdict
    score = result.score

    # Choose card style based on verdict
    if verdict == "safe":
        template = "green"
        icon = "✅"
        verdict_cn = "安全"
    elif verdict == "malicious":
        template = "red"
        icon = "🚨"
        verdict_cn = "恶意"
    else:
        template = "orange"
        icon = "⚠️"
        verdict_cn = "可疑"

    # Build evidence text
    evidence_text = ""
    if result.evidence:
        lines = []
        for i, ev in enumerate(result.evidence[:5], 1):
            lines.append(f"{i}. `{_escape(ev[:120])}`")
        evidence_text = "\n".join(lines)
    else:
        evidence_text = "无异常发现"

    # Build layer detail
    layer_text = f"Layer 1: **{result.l1_verdict.upper()}** (score {result.l1_score:.3f})"
    if result.l2_verdict:
        layer_text += f"\nLayer 2: **{result.l2_verdict.upper()}** (score {result.l2_score:.3f})"

    # Cost
    cost_text = f"${result.cost:.3f}" if result.cost > 0 else "免费"

    elements = [
        {
            "tag": "div",
            "fields": [
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": f"**技能名称**\n{_escape(result.skill_name)}"
                    }
                },
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": f"**最终判定**\n{icon} {verdict_cn} ({score:.3f})"
                    }
                }
            ]
        },
        {
            "tag": "div",
            "fields": [
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": f"**扫描层级**\n{layer_text}"
                    }
                },
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": f"**耗时 / 费用**\n{elapsed:.1f}s / {cost_text}"
                    }
                }
            ]
        },
        {"tag": "hr"},
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**检测详情**\n{evidence_text}"
            }
        },
    ]

    # Add stats
    stats_parts = [f"文件数: {result.n_scripts}"]
    if result.details.get("l1_pattern_count", 0) > 0:
        stats_parts.append(f"规则命中: {result.details['l1_pattern_count']}")
    if result.details.get("l2_skipped"):
        stats_parts.append("L2: 未配置 API Key, 跳过")

    elements.append({
        "tag": "note",
        "elements": [
            {
                "tag": "plain_text",
                "content": " | ".join(stats_parts) + " | SkillSieve"
            }
        ]
    })

    return {
        "header": {
            "template": template,
            "title": {
                "tag": "plain_text",
                "content": f"{icon} SkillSieve 扫描完成"
            }
        },
        "elements": elements,
    }


def build_error_card(error_msg: str) -> dict:
    """Build an error card."""
    return {
        "header": {
            "template": "red",
            "title": {
                "tag": "plain_text",
                "content": "❌ SkillSieve 扫描失败"
            }
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**错误信息**\n`{_escape(error_msg[:300])}`"
                }
            },
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "请检查 URL 是否正确，或联系管理员"
                    }
                ]
            }
        ]
    }


def _escape(text: str) -> str:
    """Escape special characters for Feishu markdown."""
    # Feishu lark_md uses a subset of markdown
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
