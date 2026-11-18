"""
LinkedIn post/content search tool.

Performs LinkedIn's global content search (the "Posts" results tab) using
innerText extraction, so informal "we're hiring" / "Buscamos ..." posts can
be found before a formal job listing is published. Mirrors search_people:
build a /search/results/content/ URL, scroll to load results, and return the
raw innerText for the LLM to parse, plus post-permalink references.
"""

import logging
import re
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.dependencies import get_ready_extractor, tool_errors
from linkedin_mcp_server.error_handler import CodedToolError, ErrorCode
from linkedin_mcp_server.linkedin.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    FilterValidationError,
    rate_limited_section_error,
)
from linkedin_mcp_server.linkedin.search_urls import build_content_search_url

logger = logging.getLogger(__name__)

_POST_PATH = re.compile(
    r"^/(?:posts/[A-Za-z0-9%_.\-]+"
    r"|feed/update/urn:li:(?:activity|share|ugcPost):\d{10,})/?$"
)


def normalize_post_url(post: str) -> str:
    """Return a canonical linkedin.com post permalink, or raise ValueError.

    Accepts a full or relative /posts/<slug> or /feed/update/urn:li:...:<id>
    permalink, or a bare activity id. Anything else is refused so the tool can
    never be pointed at an arbitrary page.
    """
    s = post.strip()
    if re.fullmatch(r"\d{10,}", s):
        return f"https://www.linkedin.com/feed/update/urn:li:activity:{s}/"
    s = re.sub(r"^https?://(?:[a-z]{2,3}\.|www\.)?linkedin\.com", "", s)
    s = s.split("?", 1)[0].split("#", 1)[0]
    if not _POST_PATH.match(s):
        raise ValueError(
            "post must be a LinkedIn post permalink (/posts/<slug> or "
            "/feed/update/urn:li:activity:<id>) or a bare activity id"
        )
    return "https://www.linkedin.com" + s.rstrip("/") + "/"


def register_post_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register post/content-search tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Search Posts",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"post", "search"},
    )
    @tool_errors
    async def search_posts(
        keywords: str,
        ctx: Context,
        date_posted: str | None = None,
        max_pages: Annotated[int, Field(ge=1, le=10)] = 3,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Search LinkedIn posts/content globally by keyword (the "Posts" tab).

        Use this to catch informal hiring posts ("we're hiring", "Buscamos
        ...", "estamos contratando", "join our team") that often appear before
        a formal job listing exists. This is global content search, distinct
        from get_feed (your own home feed) and get_company_posts (one
        company's page).

        Args:
            keywords: Search keywords (e.g., "Buscamos Unity", "AI automation hiring")
            ctx: FastMCP context for progress reporting
            date_posted: Optional recency filter. One of "past-24h",
                "past-week", "past-month"; the "past_24_hours" / "past_week" /
                "past_month" spellings used by search_jobs are accepted too.
                Omit for any time.
            content_type: Optional "Content type" filter. One of
                "documents" (PDFs and slide decks people upload), "images",
                "videos", "live_videos", "job_posts". Omit for all posts.
            max_pages: Scroll depth as result "pages" of ~5 scrolls each
                (1-10, default 3). Content search is an infinite scroll, so
                this caps how far the page is scrolled rather than fetching
                discrete pages.

        Returns:
            Dict with url, sections (search_results -> raw text), and optional
            references (post authors, companies, linked jobs, and kind
            "feed_post" permalinks read from the page's payload responses —
            /feed/update/<urn>/ or /posts/<slug>, both valid permalinks) and
            section_errors. The DOM carries no per-post permalink anchors;
            captured permalinks are not aligned to result order. The LLM
            should parse the raw text to extract each post's author,
            headline/role, company, body, posted date, and reaction/comment
            counts.
        """
        try:
            # Refused before the browser, as in search_people.
            build_content_search_url(
                keywords, date_posted=date_posted, content_type=content_type
            )
        except FilterValidationError as e:
            raise CodedToolError(str(e), ErrorCode.INVALID_ARGUMENT) from e

        extractor = await get_ready_extractor(ctx, tool_name="search_posts")
        logger.info(
            "Searching posts: keywords='%s', date_posted='%s', content_type='%s', "
            "max_pages=%d",
            keywords,
            date_posted,
            content_type,
            max_pages,
        )

        await ctx.report_progress(progress=0, total=100, message="Starting post search")

        try:
            result = await extractor.search_posts(
                keywords,
                date_posted=date_posted,
                content_type=content_type,
                max_pages=max_pages,
            )
        except FilterValidationError as e:
            # Validation messages carry actionable detail; surface them as
            # ToolError so mask_error_details doesn't reduce them to a
            # generic "Error calling tool 'search_posts'".
            raise CodedToolError(str(e), ErrorCode.INVALID_ARGUMENT) from e

        await ctx.report_progress(progress=100, total=100, message="Complete")

        return result

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Post Comments",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"post"},
    )
    @tool_errors
    async def get_post_comments(
        post: str,
        ctx: Context,
        max_scrolls: Annotated[int, Field(ge=1, le=10)] = 3,
    ) -> dict[str, Any]:
        """
        Read one LinkedIn post and the comments LinkedIn loads under it.

        Use when a hiring post hides its apply route in the comments ("link in
        comments", "JD below", "comment 'interested'") or gives no route at all.
        Read-only: opens the post permalink, scrolls to load comments, and
        returns the raw text. It does not expand every "Load more comments"
        or reply thread, so very long threads may be partial.

        Args:
            post: Post permalink (/posts/<slug> or
                /feed/update/urn:li:activity:<id>, full or relative) or a bare
                activity id.
            ctx: FastMCP context for progress reporting
            max_scrolls: How far to scroll to load comments (1-10, default 3).

        Returns:
            Dict with url, sections["post"] (post body followed by the loaded
            comments as raw text), optional references["post"] (commenter
            profiles and links) and section_errors. The LLM should parse the
            text for the poster's own comment, apply links, emails and DM asks.
        """
        try:
            url = normalize_post_url(post)
        except ValueError as e:
            raise CodedToolError(str(e), ErrorCode.INVALID_ARGUMENT) from e

        extractor = await get_ready_extractor(ctx, tool_name="get_post_comments")
        logger.info("Reading post comments: %s (max_scrolls=%d)", url, max_scrolls)
        await ctx.report_progress(progress=0, total=100, message="Opening post")

        extracted = await extractor.extract_page(url, "post", max_scrolls)

        result: dict[str, Any] = {"url": url, "sections": {}}
        if extracted.text and extracted.text != RATE_LIMITED_SECTION_TEXT:
            result["sections"]["post"] = extracted.text
            if extracted.references:
                result["references"] = {"post": extracted.references}
        elif extracted.text == RATE_LIMITED_SECTION_TEXT:
            result["section_errors"] = {"post": rate_limited_section_error()}
        elif extracted.error:
            result["section_errors"] = {"post": extracted.error}

        await ctx.report_progress(progress=100, total=100, message="Complete")
        return result
