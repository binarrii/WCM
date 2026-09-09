from api import handlers


def test_prompt_margins_are_removed_per_fragment_without_flattening_markdown():
    raw = """
        ## 审核要求
        - **重点要求**
          - 保留子列表的缩进
    """
    assert handlers._prompt_text(raw, """
            ## 长度
            **不超过500个中文字符**
        """) == (
        "## 审核要求\n- **重点要求**\n  - 保留子列表的缩进\n\n"
        "## 长度\n**不超过500个中文字符**"
    )
    assert raw.startswith("\n        ")  # Stored source remains indented until rendering.


def test_visual_prompts_keep_markdown_scope_and_length_requirements():
    for raw in (handlers._NSFW_SYSTEM_PROMPT, handlers._NSFW_WINDOW_SYSTEM_PROMPT):
        rendered = handlers._prompt_text(raw)
        assert rendered.startswith("## 任务")
        assert "\n## 输出要求\n" in rendered
        assert "**输出总长度不超过500个中文字符**" in rendered
        assert all(not line.startswith("    ") for line in rendered.splitlines())
