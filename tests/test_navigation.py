"""Deterministic navigation links in generated docs."""

from repodoc.agent.generate import ensure_doc_links


def test_appends_missing_links_only():
    content = "# Module\n\nSee [Child](net/http.md).\n"
    updated = ensure_doc_links(
        content,
        {
            "Child": "net/http.md",
            "Client": "net/http/Client.md",
        },
    )
    assert updated.count("net/http.md") == 1
    assert "- [Client](net/http/Client.md)" in updated
    assert "## Navigation" in updated


def test_inserts_missing_links_under_existing_heading():
    content = "## Navigation\n\n- [Old](old.md)\n\nBody\n"
    updated = ensure_doc_links(content, {"New": "new.md"})
    assert "## Navigation" in updated
    assert updated.count("## Navigation") == 1
    assert "- [New](new.md)" in updated
