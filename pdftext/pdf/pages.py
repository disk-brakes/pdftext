from __future__ import annotations

import math
import statistics
from typing import List
import unicodedata
import io

import pypdfium2 as pdfium

from pdftext.pdf.chars import get_chars, deduplicate_chars
from pdftext.pdf.utils import flatten, transform_bbox, remove_wrong_bboxes
from pdftext.schema import Blocks, Chars, Line, Lines, Pages, Span, Spans
import base64
from pdftext.schema import Page


def is_math_symbol(char):
    if len(char) != 1:
        return False

    category = unicodedata.category(char)
    return category == "Sm"


def assign_scripts(
    lines: Lines, height_threshold: float = 0.8, line_distance_threshold: float = 0.1
):
    for line in lines:
        prev_span = None
        if len(line["spans"]) < 2:
            continue

        # Skip vertical lines
        if line["bbox"].height > line["bbox"].width:
            continue

        for i, span in enumerate(line["spans"]):
            is_first = i == 0 or not prev_span["text"].strip()
            is_last = (
                i == len(line["spans"]) - 1 or not line["spans"][i + 1]["text"].strip()
            )
            span_height = span["bbox"].height
            span_top = span["bbox"].y_start
            span_bottom = span["bbox"].y_end

            line_fullheight = (
                span_height / max(1, line["bbox"].height) <= height_threshold
            )
            next_fullheight = (
                is_last
                or span_height / max(1, line["spans"][i + 1]["bbox"].height)
                <= height_threshold
            )
            prev_fullheight = (
                is_first
                or span_height / max(1, prev_span["bbox"].height) <= height_threshold
            )

            above = any(
                [
                    span_top
                    < (s["bbox"].y_start - s["bbox"].height * line_distance_threshold)
                    for j, s in enumerate(line["spans"])
                    if j != i
                ]
            )
            prev_above = is_first or span_top < prev_span["bbox"].y_start
            next_above = is_last or span_top < line["spans"][i + 1]["bbox"].y_start

            below = any(
                [
                    span_bottom
                    > (s["bbox"].y_end + s["bbox"].height * line_distance_threshold)
                    for j, s in enumerate(line["spans"])
                    if j != i
                ]
            )
            prev_below = is_first or span_bottom > prev_span["bbox"].y_end
            next_below = is_last or span_bottom > line["spans"][i + 1]["bbox"].y_end

            span_text = span["text"].strip()
            span_text_okay = all(
                [
                    (
                        len(span_text) == 1 or span_text.isdigit()
                    ),  # Ensure that the span text is a single char or a number
                    span_text.isalnum()
                    or is_math_symbol(
                        span_text
                    ),  # Ensure that the span text is an alphanumeric or a math symbol
                ]
            )

            if all(
                [
                    (prev_fullheight or next_fullheight),
                    (prev_above or next_above),
                    above,
                    line_fullheight,
                    span_text_okay,
                ]
            ):
                span["superscript"] = True
            elif all(
                [
                    (prev_fullheight or next_fullheight),
                    (prev_below or next_below),
                    below,
                    line_fullheight,
                    span_text_okay,
                ]
            ):
                span["subscript"] = True

            prev_span = span


def get_spans(
    chars: Chars,
    superscript_height_threshold: float = 0.8,
    line_distance_threshold: float = 0.1,
) -> Spans:
    spans: Spans = []
    span: Span = None
    avg_char_width = None
    sum_char_widths = 0
    prev_char_bbox = None

    def span_break():
        spans.append(
            {
                "bbox": char["bbox"],
                "text": char["char"],
                "rotation": char["rotation"],
                "font": char["font"],
                "char_start_idx": char["char_idx"],
                "char_end_idx": char["char_idx"],
                "chars": [char],
                "url": "",
            }
        )

    for char in chars:
        if spans:
            span = spans[-1]

        if not span:
            span_break()
            sum_char_widths = char["bbox"].width
            prev_char_bbox = char["bbox"]
            continue

        # we break on any change in font info
        if any(
            char["font"][k] != span["font"][k]
            for k in ["name", "flags", "size", "weight", "color"]
        ):
            span_break()
            sum_char_widths = char["bbox"].width
            prev_char_bbox = char["bbox"]
            continue

        if char["rotation"] != span["rotation"]:
            span_break()
            sum_char_widths = char["bbox"].width
            prev_char_bbox = char["bbox"]
            continue

        # we break on hyphenation or newline
        if span["text"].endswith("\x02") or span["text"].endswith("\n"):
            span_break()
            sum_char_widths = char["bbox"].width
            prev_char_bbox = char["bbox"]
            continue

        # Character is likely a superscript
        if all(
            [
                char["bbox"][1]
                < (
                    span["bbox"][1] - span["bbox"].height * line_distance_threshold
                ),  # char top is above span
                char["bbox"][3]
                < (span["bbox"].height * superscript_height_threshold)
                + span["bbox"][1],  # char bottom is not full line height
                char["bbox"][0] > span["bbox"][2],  # char is to the right of the span
            ]
        ):
            span_break()
            sum_char_widths = char["bbox"].width
            prev_char_bbox = char["bbox"]
            continue

        if prev_char_bbox:
            avg_char_width = sum_char_widths / len(span["chars"])
            if char["bbox"].horizontal_distance(prev_char_bbox) > 1.5 * avg_char_width:
                span_break()
                sum_char_widths = char["bbox"].width
                prev_char_bbox = char["bbox"]
                continue

            if char["bbox"].overlap_y(prev_char_bbox) == 0:
                span_break()
                sum_char_widths = char["bbox"].width
                prev_char_bbox = char["bbox"]
                continue

            if char["bbox"].overlap_y(span["bbox"]) < 0.1 * min(
                char["bbox"].height, span["bbox"].height
            ):
                span_break()
                sum_char_widths = char["bbox"].width
                prev_char_bbox = char["bbox"]
                continue

        span["text"] += char["char"]
        span["char_end_idx"] = char["char_idx"]
        span["bbox"] = span["bbox"].merge(char["bbox"])
        span["chars"].append(char)
        prev_char_bbox = char["bbox"]
        sum_char_widths += char["bbox"].width

    return spans


def get_lines(spans: Spans) -> Lines:
    lines: Lines = []
    line: Line = None

    def line_break():
        lines.append(
            {"spans": [span], "bbox": span["bbox"], "rotation": span["rotation"]}
        )

    for span in spans:
        if lines:
            line = lines[-1]

        if not line:
            line_break()
            continue

        # we break if the previous span ends with a linebreak or hyphenation
        if any(line["spans"][-1]["text"].endswith(suffix) for suffix in ["\n", "\x02"]):
            line_break()
            continue

        if span["rotation"] != line["rotation"]:
            line_break()
            continue

        # sometimes pdfium doesn't inject a linebreak, so we check the span positions
        if span["bbox"].y_start > line["bbox"].y_end:
            line_break()
            continue

        line["spans"].append(span)
        line["bbox"] = line["bbox"].merge(span["bbox"])

    return lines


def get_blocks(lines: Lines) -> Blocks:
    if not lines:
        return []

    x_diffs = []
    y_diffs = []
    for i in range(len(lines) - 1):
        prev_center = lines[i]["bbox"].center
        curr_center = lines[i + 1]["bbox"].center
        x_diffs.append(abs(curr_center[0] - prev_center[0]))
        y_diffs.append(abs(curr_center[1] - prev_center[1]))

    median_x_gap = 0.1
    if x_diffs:
        median_x_gap = statistics.median(x_diffs) or median_x_gap
    median_y_gap = 0.1
    if y_diffs:
        median_y_gap = statistics.median(y_diffs) or median_y_gap

    tolerance_factor = 1.5
    allowed_x_gap = median_x_gap * tolerance_factor
    allowed_y_gap = median_y_gap * tolerance_factor

    def block_merge():
        block["lines"].append(line)
        block["bbox"] = block["bbox"].merge(line["bbox"])

    blocks: Blocks = []
    for line in lines:
        if not blocks:
            # First block
            blocks.append(
                {"lines": [line], "bbox": line["bbox"], "rotation": line["rotation"]}
            )
            continue

        block = blocks[-1]
        last_line = block["lines"][-1]

        last_center = last_line["bbox"].center
        current_center = line["bbox"].center

        x_diff = abs(current_center[0] - last_center[0])
        y_diff = abs(current_center[1] - last_center[1])

        # we merge if the line is close enough to the previous line
        if x_diff <= allowed_x_gap and y_diff <= allowed_y_gap:
            block_merge()
            continue

        # we make an exception for the first line w.r.t the x diff, because the first line is usually indented
        line_x_indented_start = last_line["bbox"].x_start > line["bbox"].x_start
        if (
            len(block["lines"]) == 1
            and line_x_indented_start
            and y_diff <= allowed_y_gap
        ):
            block_merge()
            continue

        # we make an exception for the last line w.r.t the x diff, because the last line is can be incomplete
        line_x_indented_end = last_line["bbox"].x_end > line["bbox"].x_end
        if line_x_indented_end and y_diff <= allowed_y_gap:
            block_merge()
            continue

        # if the y diff is very small, and you see a line continuation, we merge (can happen with inline math between text spans)
        if (
            y_diff < allowed_y_gap * 0.2
            and last_line["bbox"].x_end > line["bbox"].x_start
        ):
            block_merge()
            continue

        # we also merge when we see the current line intersecting the previous block
        if block["bbox"].intersection_pct(line["bbox"]) > 0:
            block_merge()
            continue

        blocks.append({"lines": [line], "bbox": line["bbox"]})

    # we do one last pass of merging overlapping blocks in the PDF reading order
    merged_blocks = []
    for i in range(len(blocks)):
        if not merged_blocks:
            merged_blocks.append(blocks[i])
            continue

        prev_block = merged_blocks[-1]
        curr_block = blocks[i]

        if prev_block["bbox"].intersection_pct(curr_block["bbox"]) > 0:
            merged_blocks[-1] = {
                "lines": prev_block["lines"] + curr_block["lines"],
                "bbox": prev_block["bbox"].merge(curr_block["bbox"]),
            }
        else:
            merged_blocks.append(curr_block)

    return merged_blocks

def get_image_bboxes(
        page: pdfium.PdfPage,
        page_bbox: list[float],
        page_rotation: int
    ):

    objects = list(page.get_objects())

    text_bboxes = []
    non_text_objects = []

    for obj in objects:
        
        if obj.type in (0, 5):
            continue

        if obj.type in (2, 3, 4):
            non_text_objects.append(obj)
            continue 

        text_bboxes.append(
            transform_bbox(page_bbox, page_rotation, obj.get_pos())
        )

    for obj in objects: 

        if obj.type not in (0, 5):
            continue

        obj_bbox = transform_bbox(page_bbox, page_rotation, obj.get_pos()) 

        # check if the object bbox is intersecting any of the text bboxes
        flag = any(obj_bbox.intersection_area(text_bbox) > 0 for text_bbox in text_bboxes)

        if not flag: 
            non_text_objects.append(obj)
            continue 

    non_text_bboxes = [transform_bbox(page_bbox, page_rotation, obj.get_pos()) for obj in non_text_objects]

    # TODO - Maybe eliminate wrong bboxes
    non_text_bboxes = remove_wrong_bboxes(non_text_bboxes, page_bbox, page)

    return non_text_bboxes


def get_pages(
    pdf: pdfium.PdfDocument,
    page_range: list[int],
    flatten_pdf: bool = True,
    quote_loosebox: bool = True,
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
    page_scale: int = 2,
) -> Pages:
    pages: Pages = []

    for page_idx in page_range:
        page = pdf.get_page(page_idx)
        if flatten_pdf:
            flatten(page)
            page = pdf.get_page(page_idx)

        textpage = page.get_textpage()

        page_bbox: List[float] = page.get_bbox()
        page_width = math.ceil(abs(page_bbox[2] - page_bbox[0]))
        page_height = math.ceil(abs(page_bbox[1] - page_bbox[3]))

        page_rotation = 0
        try:
            page_rotation = page.get_rotation()
        except:
            pass

        chars = deduplicate_chars(
            get_chars(textpage, page_bbox, page_rotation, quote_loosebox)
        )
        spans = get_spans(
            chars,
            superscript_height_threshold=superscript_height_threshold,
            line_distance_threshold=line_distance_threshold,
        )
        lines = get_lines(spans)
        assign_scripts(
            lines,
            height_threshold=superscript_height_threshold,
            line_distance_threshold=line_distance_threshold,
        )
        blocks = get_blocks(lines)

        ## Adding image
        img = page.render(scale=page_scale)
        img = img.to_pil()

        images = get_image_bboxes(page, page_bbox, page_rotation)

        bytes_arr = io.BytesIO()
        try:
            img.save(bytes_arr, format="PNG")
            bytes_arr.seek(0)
            img_base64 = base64.b64encode(bytes_arr.getvalue()).decode("utf-8")

            pages.append(
                {
                    "page": page_idx,
                    "bbox": page_bbox,
                    "width": page_width,
                    "height": page_height,
                    "rotation": page_rotation,
                    "blocks": blocks,
                    "scale": page_scale,
                    "page_image": img_base64,
                    "images": images
                }
            )
        finally:
            bytes_arr.close()
            img.close()

    return pages
