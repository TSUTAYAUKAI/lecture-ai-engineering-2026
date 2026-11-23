"""第5回 宿題: 前後ページ付き検索のテスト（APIは呼ばない）。"""

import json

import pytest
from pydantic_ai.models.test import TestModel

from app.agent import NEIGHBOR_TOOL, agent
from app.retrieval import add_neighbor_pages, load_pages, search_with_neighbors


@pytest.fixture(scope="module")
def pages():
    return load_pages()


def hits_for(pages, *page_ids):
    by_id = {page["page_id"]: page for page in pages}
    return [{**by_id[page_id], "distance": 0.1} for page_id in page_ids]


def ids(results):
    return [result["page_id"] for result in results]


def test_previous_and_next_pages_follow_the_hit(pages):
    results = add_neighbor_pages(hits_for(pages, "lec02-main-p052"), pages)
    assert ids(results) == ["lec02-main-p052", "lec02-main-p051", "lec02-main-p053"]
    assert [r["role"] for r in results] == ["hit", "neighbor", "neighbor"]
    assert "distance" not in results[1] and results[1]["neighbor_of"] == "lec02-main-p052"
    assert all(r["content"] and r["title"] and r["lecture"] == 2 for r in results)


def test_neighbors_do_not_cross_documents(pages):
    last = max(p["page"] for p in pages if p["document_id"] == "lec01-main")
    results = add_neighbor_pages(hits_for(pages, f"lec01-main-p{last:03d}", "lec02-main-p001"), pages)
    assert ids(results) == [f"lec01-main-p{last:03d}", f"lec01-main-p{last - 1:03d}",
                            "lec02-main-p001", "lec02-main-p002"]


def test_pages_are_not_duplicated_and_hits_stay_hits(pages):
    # 51と52は隣同士、52と54はどちらもp.53に隣り合う
    hits = hits_for(pages, "lec02-main-p052", "lec02-main-p054", "lec02-main-p051")
    results = add_neighbor_pages(hits, pages)
    assert len(ids(results)) == len(set(ids(results)))
    assert ids([r for r in results if r["role"] == "hit"]) == ids(hits)
    p53 = next(r for r in results if r["page_id"] == "lec02-main-p053")
    assert p53["neighbor_of"] == "lec02-main-p052"


def test_search_results_keep_their_content_and_order(pages, monkeypatch):
    hits = hits_for(pages, "lec03-main-p013", "lec02-main-p052")
    monkeypatch.setattr("app.retrieval.search", lambda query: hits)
    results = [r for r in search_with_neighbors("質問") if r["role"] == "hit"]
    assert [{k: v for k, v in r.items() if k != "role"} for r in results] == hits


def test_agent_receives_neighbors_under_the_same_tool_name(pages, monkeypatch):
    monkeypatch.setattr("app.retrieval.search", lambda query: hits_for(pages, "lec02-main-p052"))
    model = TestModel(call_tools=["search_materials"])
    with agent.override(model=model, tools=[NEIGHBOR_TOOL]):
        result = agent.run_sync("LoRAとは？")
    offered = [t.name for t in model.last_model_request_parameters.function_tools]
    assert offered == ["search_materials"]
    returned = [p for m in result.all_messages() for p in m.parts if p.part_kind == "tool-return"]
    assert len(json.loads(returned[0].content)["results"]) == 3
