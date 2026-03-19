# AI 粤语填词改编 Agent

> COM6104 2025/26 Semester 2 Group Project

[English](README.md) | [中文版](README.zh-CN.md)

## 📖 项目简介

本项目是一个基于 Agentic AI 的粤语填词改编系统。主用途是把外语歌或现有歌词改编成可唱的粤语版本；如果用户没有提供原歌词，也可以只给主题、情景或意境文本，让系统进行原创填词。输入为 MIDI 旋律与来源歌词/主题文本，系统会分析旋律音节结构，并生成符合粤语发音、0243 旋律贴合度与押韵规则的新歌词。

## 环境配置

项目现在会自动读取仓库根目录下的 `.env`。推荐做法：

```powershell
Copy-Item .env.example .env
```

常用默认值：

- `LLM_PROVIDER=lmstudio`
- `LMSTUDIO_MODEL=qwen3.5-4b@q4_k_m`
- `OLLAMA_MODEL=qwen3.5:4b`
