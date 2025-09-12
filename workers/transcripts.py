"""Transcript PDF extraction and graph ingestion (MVP).

This module implements:
- Extract transcript utterances from a PDF stored in MinIO
- Persist utterances.jsonl + report.json to MinIO
- Upsert utterances to Postgres (utterance table)
- Ingest Call/Speaker/Utterance graph into Neo4j

No local ML; library-based extraction + deterministic heuristics only.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import structlog

from core.settings import SETTINGS
from infra.resources import DatabaseResource, MinIOResource, Neo4jResource
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.documents.entities.document import Document as DocumentEntity
from api.features.query.entities.utterance import Utterance as UtteranceEntity
from api.features.query.entities.chunk import (
    Chunk as ChunkEntity,
    ChunkType as ChunkTypeEnum,
)

logger = structlog.get_logger("workers.transcripts")


@dataclass
class Page:
    page_no: int
    text: str
    extraction_method: str = "text"


@dataclass
class Utterance:
    utterance_id: str
    doc_id: str
    turn_index: int
    speaker: str
    role: str
    speech: str
    section: str
    page_spans: List[Dict[str, Any]]
    extraction_method: str
    meta: Dict[str, Any]


# --------- Storage Keys ---------


def utterances_key(doc_id: str) -> str:
    return f"transcripts/{doc_id}/utterances.jsonl"


def report_key(doc_id: str) -> str:
    return f"transcripts/{doc_id}/report.json"


# --------- PDF Extraction ---------


def _normalize_text(text: str) -> str:
    # Merge hyphenated line breaks: word-\n continuation -> wordcontinuation
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace newlines that are mid-paragraph with spaces when not followed by empty line
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize em dashes and colon spacing
    text = text.replace("—", " - ")
    text = re.sub(r"\s*:\s*", ": ", text)
    # Preserve paragraph breaks as double newline
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _extract_pages_with_pymupdf(pdf_bytes: bytes) -> List[Page]:
    import fitz  # type: ignore

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[Page] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            pages.append(Page(page_no=i + 1, text=text, extraction_method="text"))
    finally:
        doc.close()
    return pages


def _extract_pages_with_pdfminer(pdf_bytes: bytes) -> List[Page]:
    from pdfminer.high_level import extract_text  # type: ignore

    # pdfminer works on file-like objects
    text = extract_text(io.BytesIO(pdf_bytes))
    # It returns full doc text; split naively by form feed (page breaks) if present
    # Many PDFs may not include FF; in that case, produce single page
    parts = text.split("\x0c") if "\x0c" in text else [text]
    pages: List[Page] = []
    for idx, part in enumerate(parts, start=1):
        pages.append(Page(page_no=idx, text=part, extraction_method="text"))
    return pages


def extract_pages(pdf_bytes: bytes) -> List[Page]:
    """Try PyMuPDF then pdfminer; no OCR in MVP."""
    try:
        return _extract_pages_with_pymupdf(pdf_bytes)
    except Exception as e:
        logger.warning("pymupdf_failed_fallback_pdfminer", error=str(e))
        try:
            return _extract_pages_with_pdfminer(pdf_bytes)
        except Exception as e2:
            logger.error("pdfminer_failed", error=str(e2))
            raise RuntimeError(
                "Failed to extract text from PDF. Install extras: pip install '.[pdf]'"
            )


# --------- Speaker Segmentation ---------

SPEAKER_LABEL_RE = re.compile(
    r"^(?P<label>(?:[A-Z][A-Za-z'\-\.]+(?:\s+[A-Z][A-Za-z'\-\.]+)*|[A-Z][A-Z ]+)(?:\s*\([^)]*\))?(?:\s*—\s*[^:]+)?)\s*:\s*(?P<rest>.*)$"
)
SECTION_HEADERS = {
    "participants": re.compile(r"^participants$", re.IGNORECASE),
    "prepared_remarks": re.compile(
        r"^(prepared\s+remarks|opening\s+remarks)$", re.IGNORECASE
    ),
    "qa": re.compile(r"^(q\s*&\s*a|question\s+and\s+answer|q\+a)$", re.IGNORECASE),
}


def _infer_role(label: str) -> Tuple[str, Dict[str, Any]]:
    lbl = label.strip()
    norm = lbl.lower()
    meta: Dict[str, Any] = {}
    if "operator" in norm:
        return "operator", meta
    # Analysts commonly have em dash with firm
    if "—" in lbl or " - " in lbl:
        parts = re.split(r"\s*[—-]\s*", lbl, maxsplit=1)
        if len(parts) == 2:
            meta["firm"] = parts[1]
        return "analyst", meta
    # Titles hint at management
    if re.search(
        r"\b(ceo|cfo|coo|chief|president|vp|officer|founder|head|director)\b", norm
    ):
        return "management", meta
    return "unknown", meta


def segment_utterances(
    pages: List[Page], doc_id: str
) -> Tuple[List[Utterance], Dict[str, Any]]:
    current_section = "other"
    turns: List[Utterance] = []
    turn_index = 0
    current_speaker = None
    current_role = "unknown"
    current_meta: Dict[str, Any] = {}
    current_text_parts: List[str] = []
    current_page_spans: List[Dict[str, Any]] = []

    def flush_turn():
        nonlocal \
            turn_index, \
            current_text_parts, \
            current_page_spans, \
            current_speaker, \
            current_role, \
            current_meta
        if current_speaker and current_text_parts:
            speech = _normalize_text(" ".join(current_text_parts)).strip()
            if speech:
                uid_src = f"{doc_id}|{current_speaker}|{speech}|{turn_index}"
                uid = hashlib.sha256(uid_src.encode("utf-8")).hexdigest()
                turns.append(
                    Utterance(
                        utterance_id=uid,
                        doc_id=doc_id,
                        turn_index=turn_index,
                        speaker=current_speaker,
                        role=current_role,
                        speech=speech,
                        section=current_section,
                        page_spans=current_page_spans.copy(),
                        extraction_method="text",
                        meta=current_meta.copy(),
                    )
                )
                turn_index += 1
        # reset accumulators
        current_text_parts = []
        current_page_spans = []

    for page in pages:
        # Cheap header/footer removal: drop extremely short lines that repeat page number patterns
        lines = [ln.strip() for ln in page.text.splitlines()]
        for raw in lines:
            if not raw:
                continue
            # Section headers
            if SECTION_HEADERS["participants"].match(raw):
                flush_turn()
                current_section = "participants"
                continue
            if SECTION_HEADERS["prepared_remarks"].match(raw):
                flush_turn()
                current_section = "prepared_remarks"
                continue
            if SECTION_HEADERS["qa"].match(raw):
                flush_turn()
                current_section = "qa"
                continue

            m = SPEAKER_LABEL_RE.match(raw)
            if m:
                # Start new turn
                flush_turn()
                label = m.group("label").strip()
                rest = m.group("rest").strip()
                role, meta = _infer_role(label)
                current_speaker = re.sub(r"\s{2,}", " ", label)
                current_role = role
                current_meta = meta
                if rest:
                    current_text_parts.append(rest)
                current_page_spans.append({"page_no": page.page_no})
            else:
                # Continuation of current speaker or unmatched text (skip if no speaker yet)
                if current_speaker:
                    current_text_parts.append(raw)
                    # Only add page once per contiguous block; here we append only if empty
                    if (
                        not current_page_spans
                        or current_page_spans[-1].get("page_no") != page.page_no
                    ):
                        current_page_spans.append({"page_no": page.page_no})
                else:
                    # Unmatched lines before first speaker: ignore or collect to meta
                    pass
        # end for lines
    # end for pages

    # Final flush
    flush_turn()

    report = {
        "pages_total": len(pages),
        "turns_total": len(turns),
        "speakers_distinct": len({t.speaker for t in turns}),
        "sections": {
            s: sum(1 for t in turns if t.section == s)
            for s in ["participants", "prepared_remarks", "qa", "other"]
        },
        "extraction_method": "text",
    }
    return turns, report


# --------- MinIO I/O ---------


async def _get_minio() -> MinIOResource:
    minio = MinIOResource(
        endpoint=SETTINGS.MINIO.MINIO_ENDPOINT,
        access_key=SETTINGS.MINIO.MINIO_ACCESS_KEY,
        secret_key=SETTINGS.MINIO.MINIO_SECRET_KEY.get_secret_value(),
        bucket_name=SETTINGS.MINIO.MINIO_BUCKET,
    )
    await minio.init()
    return minio


async def _get_db() -> DatabaseResource:
    db = DatabaseResource(database_url=str(SETTINGS.DATABASE.DATABASE_URL))
    await db.init()
    return db


async def _get_neo4j() -> Neo4jResource:
    neo = Neo4jResource(
        uri=SETTINGS.NEO4J.NEO4J_URI,
        user=SETTINGS.NEO4J.NEO4J_USER,
        password=SETTINGS.NEO4J.NEO4J_PASSWORD.get_secret_value(),
    )
    await neo.init()
    return neo


async def read_pdf_from_minio(doc: DocumentEntity, minio: MinIOResource) -> bytes:
    """Download raw PDF bytes from MinIO given a Document row."""
    client = minio.client
    bucket = minio.bucket_name
    resp = client.get_object(bucket, doc.raw_path)
    try:
        data = resp.read()
        return data
    finally:
        resp.close()
        resp.release_conn()


async def write_jsonl_to_minio(
    doc_id: str, items: Iterable[Dict[str, Any]], minio: MinIOResource
) -> str:
    key = utterances_key(doc_id)
    payload = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode(
        "utf-8"
    )
    client = minio.client
    bucket = minio.bucket_name
    client.put_object(
        bucket,
        key,
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/jsonl",
    )
    return key


async def write_report_to_minio(
    doc_id: str, report: Dict[str, Any], minio: MinIOResource
) -> str:
    key = report_key(doc_id)
    payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    client = minio.client
    bucket = minio.bucket_name
    client.put_object(
        bucket,
        key,
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )
    return key


# --------- Postgres Upsert ---------


async def upsert_utterances_pg(
    doc_id: str, utterances: List[Utterance], session: AsyncSession
) -> int:
    # Load existing ids to avoid duplicates
    existing_stmt = select(UtteranceEntity.utterance_id).where(
        UtteranceEntity.document_id == doc_id
    )
    res = await session.execute(existing_stmt)
    existing_ids = set(res.scalars().all())

    added = 0
    for u in utterances:
        if u.utterance_id in existing_ids:
            continue
        row = UtteranceEntity(
            document_id=doc_id,
            utterance_id=u.utterance_id,
            turn_index=u.turn_index,
            speaker=u.speaker,
            role=u.role,
            section=u.section,
            speech=u.speech,
            page_spans=u.page_spans,
            extraction_method=u.extraction_method,
        )
        session.add(row)
        added += 1
    await session.commit()
    return added


# --------- Neo4j Ingestion ---------


def _ensure_neo4j_constraints(session) -> None:
    session.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Call) REQUIRE c.doc_id IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (u:Utterance) REQUIRE u.utterance_id IS UNIQUE"
    )
    session.run("CREATE INDEX IF NOT EXISTS FOR (s:Speaker) ON (s.name)")


def ingest_graph_neo4j(
    doc_id: str, utterances: List[Utterance], neo: Neo4jResource
) -> int:
    driver = neo.driver
    assert driver is not None
    count = 0
    with driver.session() as s:
        _ensure_neo4j_constraints(s)
        # Upsert Call
        s.run(
            "MERGE (c:Call {doc_id:$doc_id}) ON CREATE SET c.source=$source",
            doc_id=doc_id,
            source=f"minio://{SETTINGS.MINIO.MINIO_BUCKET}/raw/{doc_id}",
        )
        # Upsert Speakers roster
        speakers = {u.speaker: u for u in utterances}
        for spk, u in speakers.items():
            s.run(
                "MERGE (s:Speaker {name:$name, firm:$firm})",
                name=spk,
                firm=u.meta.get("firm"),
            )
        # Utterances + relationships
        prev_uid: Optional[str] = None
        for u in utterances:
            s.run(
                "MERGE (u:Utterance {utterance_id:$uid}) "
                "ON CREATE SET u.turn_index=$turn, u.section=$section",
                uid=u.utterance_id,
                turn=u.turn_index,
                section=u.section,
            )
            s.run(
                "MATCH (c:Call {doc_id:$doc_id}), (u:Utterance {utterance_id:$uid}) "
                "MERGE (c)-[:CONTAINS]->(u)",
                doc_id=doc_id,
                uid=u.utterance_id,
            )
            s.run(
                "MATCH (s:Speaker {name:$name}), (u:Utterance {utterance_id:$uid}) "
                "MERGE (s)-[:SPOKE]->(u)",
                name=u.speaker,
                uid=u.utterance_id,
            )
            if prev_uid is not None:
                s.run(
                    "MATCH (u1:Utterance {utterance_id:$u1}), (u2:Utterance {utterance_id:$u2}) "
                    "MERGE (u1)-[:NEXT]->(u2)",
                    u1=prev_uid,
                    u2=u.utterance_id,
                )
            prev_uid = u.utterance_id
            count += 1
    return count


# --------- Orchestration API ---------


async def process_transcript_document(doc_id: str) -> Dict[str, Any]:
    """End-to-end: MinIO PDF -> utterances.jsonl -> Postgres + Neo4j."""
    # Resources
    minio, db = await asyncio.gather(_get_minio(), _get_db())

    # Fetch document
    async with db.get_session() as session:
        stmt = select(DocumentEntity).where(DocumentEntity.id == doc_id)
        res = await session.execute(stmt)
        doc = res.scalar_one_or_none()
        if not doc:
            raise RuntimeError(f"Document {doc_id} not found")
        # quick guard: ensure doc_type is transcript
        doc_type_val = str(getattr(doc, "doc_type", "")).lower()
        if "transcript" not in doc_type_val:
            raise RuntimeError(
                f"Document {doc_id} is not a transcript (doc_type={doc_type_val})"
            )

    # Download
    pdf_bytes = await read_pdf_from_minio(doc, minio)

    # Extract
    pages = extract_pages(pdf_bytes)
    for p in pages:
        p.text = _normalize_text(p.text)

    # Segment
    turns, report = segment_utterances(pages, doc_id)

    # Persist artifacts
    key_jsonl, key_report = await asyncio.gather(
        write_jsonl_to_minio(doc_id, [u.__dict__ for u in turns], minio),
        write_report_to_minio(doc_id, report, minio),
    )

    # Postgres upsert and document metadata update
    async with db.get_session() as session:
        inserted = await upsert_utterances_pg(doc_id, turns, session)
        # Update Document row: attach artifact keys to processing_metadata
        stmt = select(DocumentEntity).where(DocumentEntity.id == doc_id)
        res = await session.execute(stmt)
        ent = res.scalar_one_or_none()
        if ent:
            meta = ent.processing_metadata or {}
            meta.update(
                {
                    "utterances_key": key_jsonl,
                    "report_key": key_report,
                    "utterances_count": len(turns),
                }
            )
            ent.processing_metadata = meta
            await session.commit()

    # Neo4j ingest
    neo = await _get_neo4j()
    ingested = ingest_graph_neo4j(doc_id, turns, neo)

    return {
        "doc_id": doc_id,
        "utterances_written": len(turns),
        "pg_inserted": inserted,
        "neo4j_ingested": ingested,
        "utterances_key": key_jsonl,
        "report_key": key_report,
    }


async def create_utterances_jsonl(doc_id: str) -> Dict[str, Any]:
    """Extract + segment transcript and write MinIO artifacts only.

    Sets Document.status to CHUNKED and records artifact keys in processing_metadata.
    """
    minio, db = await asyncio.gather(_get_minio(), _get_db())

    # Load document
    async with db.get_session() as session:
        stmt = select(DocumentEntity).where(DocumentEntity.id == doc_id)
        res = await session.execute(stmt)
        doc = res.scalar_one_or_none()
        if not doc:
            raise RuntimeError(f"Document {doc_id} not found")
        if "transcript" not in str(getattr(doc, "doc_type", "")).lower():
            raise RuntimeError(f"Document {doc_id} is not a transcript")

    pdf_bytes = await read_pdf_from_minio(doc, minio)
    pages = extract_pages(pdf_bytes)
    for p in pages:
        p.text = _normalize_text(p.text)
    turns, report = segment_utterances(pages, doc_id)

    key_jsonl, key_report = await asyncio.gather(
        write_jsonl_to_minio(doc_id, [u.__dict__ for u in turns], minio),
        write_report_to_minio(doc_id, report, minio),
    )

    # Metadata update only
    async with db.get_session() as session:
        stmt = select(DocumentEntity).where(DocumentEntity.id == doc_id)
        res = await session.execute(stmt)
        ent = res.scalar_one_or_none()
        if ent:
            meta = ent.processing_metadata or {}
            meta.update(
                {
                    "utterances_key": key_jsonl,
                    "report_key": key_report,
                    "utterances_count": len(turns),
                }
            )
            ent.processing_metadata = meta
            await session.commit()

    return {
        "doc_id": doc_id,
        "utterances_written": len(turns),
        "utterances_key": key_jsonl,
        "report_key": key_report,
    }


async def load_utterances_from_minio(
    doc_id: str, minio: MinIOResource
) -> List[Utterance]:
    """Read utterances.jsonl from MinIO and parse to Utterance dataclasses."""
    client = minio.client
    bucket = minio.bucket_name
    key = utterances_key(doc_id)
    resp = client.get_object(bucket, key)
    utterances: List[Utterance] = []
    try:
        buffer = ""
        for chunk in resp.stream(32 * 1024):  # bytes
            buffer += chunk.decode("utf-8")
            while True:
                nl = buffer.find("\n")
                if nl == -1:
                    break
                line = buffer[:nl]
                buffer = buffer[nl + 1 :]
                if not line:
                    continue
                obj = json.loads(line)
                utterances.append(
                    Utterance(
                        utterance_id=obj["utterance_id"],
                        doc_id=obj["doc_id"],
                        turn_index=obj["turn_index"],
                        speaker=obj["speaker"],
                        role=obj.get("role", "unknown"),
                        speech=obj["speech"],
                        section=obj.get("section", "other"),
                        page_spans=obj.get("page_spans", []),
                        extraction_method=obj.get("extraction_method", "text"),
                        meta=obj.get("meta", {}),
                    )
                )
        # Handle trailing buffer without newline
        tail = buffer.strip()
        if tail:
            obj = json.loads(tail)
            utterances.append(
                Utterance(
                    utterance_id=obj["utterance_id"],
                    doc_id=obj["doc_id"],
                    turn_index=obj["turn_index"],
                    speaker=obj["speaker"],
                    role=obj.get("role", "unknown"),
                    speech=obj["speech"],
                    section=obj.get("section", "other"),
                    page_spans=obj.get("page_spans", []),
                    extraction_method=obj.get("extraction_method", "text"),
                    meta=obj.get("meta", {}),
                )
            )
    finally:
        resp.close()
        resp.release_conn()
    return utterances


async def ingest_transcript_pg_from_minio(doc_id: str) -> Dict[str, Any]:
    """Load utterances.jsonl from MinIO and upsert into Postgres."""
    minio, db = await asyncio.gather(_get_minio(), _get_db())
    turns = await load_utterances_from_minio(doc_id, minio)
    async with db.get_session() as session:
        inserted = await upsert_utterances_pg(doc_id, turns, session)
    return {"doc_id": doc_id, "pg_inserted": inserted}


async def ingest_transcript_neo4j_from_minio(doc_id: str) -> Dict[str, Any]:
    """Load utterances.jsonl from MinIO and ingest graph into Neo4j."""
    minio = await _get_minio()
    turns = await load_utterances_from_minio(doc_id, minio)
    neo = await _get_neo4j()
    ingested = ingest_graph_neo4j(doc_id, turns, neo)
    return {"doc_id": doc_id, "neo4j_ingested": ingested}


async def materialize_transcript_chunks_from_pg(doc_id: str) -> Dict[str, Any]:
    """Create Chunk rows (one per utterance) to integrate with generic embedding pipeline."""
    db = await _get_db()
    inserted = 0
    async with db.get_session() as session:
        # Load utterances for document
        u_stmt = (
            select(UtteranceEntity)
            .where(UtteranceEntity.document_id == doc_id)
            .order_by(UtteranceEntity.turn_index.asc())
        )
        res = await session.execute(u_stmt)
        utterances: List[UtteranceEntity] = list(res.scalars().all())
        if not utterances:
            return {"doc_id": doc_id, "chunks_inserted": 0}

        # Existing chunk hashes to avoid dupes
        # We compute hash as sha256(doc_id|turn_index|speaker|speech)
        existing_hashes: set[str] = set()
        # To reduce queries, fetch existing chunk_hash for this doc
        from sqlalchemy import select as sa_select

        ex_stmt = sa_select(ChunkEntity.chunk_hash).where(
            ChunkEntity.document_id == doc_id
        )
        ex_res = await session.execute(ex_stmt)
        existing_hashes = set(ex_res.scalars().all())

        seq = 0
        for u in utterances:
            content = u.speech or ""
            content_norm = _normalize_text(content)
            seq = u.turn_index
            h_src = f"{doc_id}|{u.turn_index}|{u.speaker}|{content_norm}"
            chash = hashlib.sha256(h_src.encode("utf-8")).hexdigest()
            if chash in existing_hashes:
                continue
            # Approximate token count by whitespace tokens (MVP)
            token_count = max(1, len(content_norm.split()))
            chunk = ChunkEntity(
                document_id=doc_id,
                chunk_hash=chash,
                chunk_type=ChunkTypeEnum.TEXT,
                content=content,
                content_normalized=content_norm,
                page_number=None,
                section_title=u.section or None,
                sequence_number=seq,
                token_count=token_count,
                extra_metadata={
                    "speaker": u.speaker,
                    "role": u.role,
                    "utterance_id": u.utterance_id,
                },
            )
            session.add(chunk)
            inserted += 1
        await session.commit()
    return {"doc_id": doc_id, "chunks_inserted": inserted}
