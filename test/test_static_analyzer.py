import sys
import os
import tempfile
import textwrap

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)

from src.cli.analyze import StaticAnalyzer


def test_analyze_file_finds_openai_chat_completion_call():
    code = textwrap.dedent("""
    from openai import OpenAI

    def run():
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":"Hello there"}],
            temperature=0.2,
            max_tokens=123
        )
        return r
    """)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sample.py")
        with open(path, "w") as f:
            f.write(code)

        sa = StaticAnalyzer()
        calls = sa.analyze_file(path)

        print("Calls:", calls)
        assert len(calls) == 1

        call = calls[0]
        assert call["model"] == "gpt-4o-mini"
        assert call["prompt"] == "Hello there"
        assert call["temperature"] == 0.2
        assert call["max_tokens"] == 123
        assert call["file"] == path
        assert isinstance(call["line"], int)


def test_analyze_file_detects_fstring_prompt_pattern():
    code = textwrap.dedent("""
    from openai import OpenAI

    def run(x):
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Hello {x}"}],
        )
        return r
    """)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sample2.py")
        with open(path, "w") as f:
            f.write(code)

        sa = StaticAnalyzer()
        calls = sa.analyze_file(path)

        print("Calls:", calls)
        assert len(calls) == 1

        call = calls[0]
        assert call["model"] == "gpt-4o-mini"
        assert call["prompt"] in ("<f-string>", "<dynamic>")
        assert call["prompt_pattern"] in ("Hello {...}", None)


if __name__ == "__main__":
    test_analyze_file_finds_openai_chat_completion_call()
    test_analyze_file_detects_fstring_prompt_pattern()
    print("\nALL STATIC ANALYZER TESTS PASSED")
