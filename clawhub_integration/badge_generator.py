"""SVG security badge generator for SkillSieve scan results."""

def generate_badge(verdict: str, confidence: float, layer: int) -> str:
    """Generate an SVG badge showing scan result.

    Args:
        verdict: "safe", "suspicious", or "malicious"
        confidence: 0.0-1.0
        layer: which layer the scan stopped at (1-3)

    Returns:
        SVG string
    """
    colors = {
        "safe": "#4c1",       # green
        "suspicious": "#fe7d37",  # orange
        "malicious": "#e05d44",   # red
    }
    labels = {
        "safe": "SAFE",
        "suspicious": "SUSPICIOUS",
        "malicious": "MALICIOUS",
    }

    color = colors.get(verdict, "#9f9f9f")
    label = labels.get(verdict, "UNKNOWN")
    conf_pct = f"{confidence * 100:.0f}%"

    # Standard shields.io-style badge SVG
    left_text = "SkillSieve"
    right_text = f"{label} | L{layer} {conf_pct}"

    left_width = len(left_text) * 6.5 + 12
    right_width = len(right_text) * 6.5 + 12
    total_width = left_width + right_width

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width:.0f}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="a">
    <rect width="{total_width:.0f}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#a)">
    <rect width="{left_width:.0f}" height="20" fill="#555"/>
    <rect x="{left_width:.0f}" width="{right_width:.0f}" height="20" fill="{color}"/>
    <rect width="{total_width:.0f}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,sans-serif" font-size="11">
    <text x="{left_width / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{left_text}</text>
    <text x="{left_width / 2:.0f}" y="14">{left_text}</text>
    <text x="{left_width + right_width / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{right_text}</text>
    <text x="{left_width + right_width / 2:.0f}" y="14">{right_text}</text>
  </g>
</svg>'''
    return svg
