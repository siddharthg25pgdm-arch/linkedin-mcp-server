import pytest

from linkedin_mcp_server.tools.post import normalize_post_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "7528682246627672064",
            "https://www.linkedin.com/feed/update/urn:li:activity:7528682246627672064/",
        ),
        (
            "/feed/update/urn:li:activity:7528682246627672064/",
            "https://www.linkedin.com/feed/update/urn:li:activity:7528682246627672064/",
        ),
        (
            "https://in.linkedin.com/posts/geetika-saha_hiring-share-7528682245943963648-_Gc6?utm_source=x",
            "https://www.linkedin.com/posts/geetika-saha_hiring-share-7528682245943963648-_Gc6/",
        ),
        (
            "https://www.linkedin.com/feed/update/urn:li:ugcPost:7528849058975596546",
            "https://www.linkedin.com/feed/update/urn:li:ugcPost:7528849058975596546/",
        ),
    ],
)
def test_normalize_post_url_accepts_permalinks(raw, expected):
    assert normalize_post_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "https://evil.example/posts/x",
        "/in/someone/",
        "/feed/",
        "https://www.linkedin.com/jobs/view/123",
        "123",
    ],
)
def test_normalize_post_url_refuses_other_pages(raw):
    with pytest.raises(ValueError):
        normalize_post_url(raw)
