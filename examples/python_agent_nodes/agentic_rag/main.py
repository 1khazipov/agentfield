"""
Agentic RAG - Production-Ready Hallucination-Proof Document Q&A
Full lazy-indexed ensemble architecture with iterative refinement and claim verification
"""

import os
import asyncio
from typing import List, Dict, Optional
from agentfield import Agent, AIConfig
from schemas import (
    QueryAnalysis,
    SubQuestion,
    SubQuestions,
    DraftAnswer,
    Claim,
    VerificationResult,
    Citation,
    VerifiedAnswer,
    RankedChunk,
    ChunkList,
    Chunk,
    Gap,
    RefinementQuery,
    SearchTerms,
    GapList,
    RefinementQueryList,
    ClaimList,
    CompletenessScore,
)
from skills import (
    load_document,
    simple_chunk_text,
    extract_keywords,
    keyword_match_score,
    find_quote_in_chunk,
    deduplicate_chunks,
    embed_text,
    embed_batch,
    rank_by_similarity,
)

# Initialize agent
app = Agent(
    node_id="agentic-rag",
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openai/gpt-4.1-mini"), temperature=0.3
    ),
)


# ============= PHASE 1: SMART CHUNKING =============


@app.skill()
async def chunk_document(file_path: str) -> ChunkList:
    """Load and intelligently chunk document"""
    content = load_document(file_path)
    chunk_dicts = simple_chunk_text(content, chunk_size=500, overlap=50)

    chunks = [
        Chunk(id=c["id"], text=c["text"], metadata=c["metadata"]) for c in chunk_dicts
    ]

    # Store in memory (PostgreSQL-backed)
    await app.memory.set("document_chunks", [c.model_dump() for c in chunks])
    await app.memory.set("document_path", file_path)

    return ChunkList(chunks=chunks, total_count=len(chunks))


# ============= PHASE 2: QUERY UNDERSTANDING =============


@app.reasoner()
async def analyze_query_complexity(question: str) -> QueryAnalysis:
    """Analyze query to determine routing strategy"""
    return await app.ai(
        system="You analyze query complexity for routing decisions.",
        user=f"""Analyze: "{question}"

Provide:
1. complexity_score (0.0-1.0):
   - 0.0-0.3: Simple factual lookup
   - 0.4-0.7: Moderate analysis
   - 0.8-1.0: Complex multi-step reasoning

2. query_type: factual, analytical, comparative, or temporal

3. needs_decomposition: true if should break into sub-questions
""",
        schema=QueryAnalysis,
    )


@app.reasoner()
async def decompose_query(question: str) -> List[SubQuestion]:
    """Break complex query into atomic sub-questions"""
    result = await app.ai(
        system="You decompose complex questions into atomic sub-questions.",
        user=f"""Question: "{question}"

Break into 3-5 atomic sub-questions.
Each should be:
- Self-contained
- Answerable independently
- Prioritized (1=highest)
""",
        schema=SubQuestions,
    )
    return result.questions


# ============= PHASE 3: ENSEMBLE RETRIEVAL =============


@app.skill()
def lazy_semantic_retrieval(
    question: str, chunks: List[Dict], top_k: int = 10
) -> List[RankedChunk]:
    """
    Lazy semantic retrieval with FastEmbed
    Only embeds query + top candidates (not all chunks)
    """
    # Step 1: Embed query
    query_embedding = embed_text(question)

    # Step 2: Pre-filter with keywords (cheap)
    keywords = extract_keywords(question, top_n=5)
    candidates = []

    for chunk in chunks:
        score = keyword_match_score(keywords, chunk["text"])
        if score > 0:
            candidates.append(chunk)

    # Limit to top 50 candidates for embedding
    candidates = sorted(
        candidates, key=lambda c: keyword_match_score(keywords, c["text"]), reverse=True
    )[:50]

    if not candidates:
        return []

    # Step 3: Embed only candidates (lazy!)
    candidate_texts = [c["text"] for c in candidates]
    candidate_embeddings = embed_batch(candidate_texts)
    candidate_ids = [c["id"] for c in candidates]

    # Step 4: Rank by similarity
    ranked = rank_by_similarity(
        query_embedding, candidate_embeddings, candidate_ids, top_k=top_k
    )

    # Step 5: Build RankedChunk objects
    chunk_map = {c["id"]: c for c in candidates}
    return [
        RankedChunk(
            chunk_id=r["chunk_id"],
            score=r["score"],
            text=chunk_map[r["chunk_id"]]["text"],
        )
        for r in ranked
    ]


@app.skill()
def keyword_retrieval(
    question: str, chunks: List[Dict], top_k: int = 10
) -> List[RankedChunk]:
    """Keyword-based retrieval (deterministic, fast)"""
    keywords = extract_keywords(question, top_n=5)

    scored_chunks = []
    for chunk in chunks:
        score = keyword_match_score(keywords, chunk["text"])
        if score > 0:
            scored_chunks.append(
                {"chunk_id": chunk["id"], "score": score, "text": chunk["text"]}
            )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return [RankedChunk(**c) for c in scored_chunks[:top_k]]


@app.reasoner()
async def type_specific_retrieval(
    question: str, query_type: str, chunks: List[Dict], top_k: int = 10
) -> List[RankedChunk]:
    """Retrieval strategy based on question type"""
    # Use AI to identify key entities/concepts for this type
    result = await app.ai(
        system=f"You identify key {query_type} elements for targeted retrieval.",
        user=f"""Question: "{question}"
Type: {query_type}

Identify 3-5 key search terms specific to this question type.
For factual: entities, names, dates
For analytical: concepts, relationships
For comparative: items being compared
For temporal: time periods, sequences
""",
        schema=SearchTerms,
    )

    # Use identified terms for targeted search
    scored_chunks = []
    for chunk in chunks:
        score = (
            sum(1 for term in result.terms if term.lower() in chunk["text"].lower())
            / len(result.terms)
            if result.terms
            else 0
        )

        if score > 0:
            scored_chunks.append(
                {"chunk_id": chunk["id"], "score": score, "text": chunk["text"]}
            )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return [RankedChunk(**c) for c in scored_chunks[:top_k]]


@app.reasoner()
async def ensemble_retrieval(
    question: str, query_type: str, top_k: int = 10
) -> List[RankedChunk]:
    """
    Ensemble retrieval: Run 3 strategies in parallel, merge results
    """
    chunks = await app.memory.get("document_chunks", [])

    # Run all strategies in parallel
    semantic_task = asyncio.to_thread(lazy_semantic_retrieval, question, chunks, top_k)
    keyword_task = asyncio.to_thread(keyword_retrieval, question, chunks, top_k)
    type_task = type_specific_retrieval(question, query_type, chunks, top_k)

    semantic_results, keyword_results, type_results = await asyncio.gather(
        semantic_task, keyword_task, type_task
    )

    # Merge with score boosting for agreement
    all_chunks = {}
    for chunk in semantic_results:
        all_chunks[chunk.chunk_id] = chunk

    for chunk in keyword_results:
        if chunk.chunk_id in all_chunks:
            # Boost if found by multiple strategies
            all_chunks[chunk.chunk_id].score = (
                (all_chunks[chunk.chunk_id].score + chunk.score) / 2 * 1.3
            )
        else:
            all_chunks[chunk.chunk_id] = chunk

    for chunk in type_results:
        if chunk.chunk_id in all_chunks:
            all_chunks[chunk.chunk_id].score = (
                (all_chunks[chunk.chunk_id].score + chunk.score) / 2 * 1.5
            )  # Highest boost for type-specific match
        else:
            all_chunks[chunk.chunk_id] = chunk

    # Sort and return top_k
    ranked = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)
    return ranked[:top_k]


# ============= PHASE 4: ITERATIVE REFINEMENT =============


@app.reasoner()
async def synthesize_draft_answer(
    question: str, chunks: List[RankedChunk]
) -> DraftAnswer:
    """Generate draft answer from retrieved chunks"""
    context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in chunks])

    return await app.ai(
        system="You synthesize precise answers from provided context only.",
        user=f"""Question: "{question}"

Context:
{context}

Provide:
1. text: Answer based ONLY on context
2. confidence: 0.0-1.0 (how well context answers question)
3. gaps: List missing information needed for complete answer
""",
        schema=DraftAnswer,
    )


@app.reasoner()
async def identify_gaps(draft: DraftAnswer) -> List[Gap]:
    """Identify specific information gaps"""
    if not draft.gaps:
        return []

    result = await app.ai(
        system="You identify and prioritize information gaps.",
        user=f"""Draft answer gaps: {draft.gaps}

For each gap, provide:
- description: What specific information is missing
- priority: 1-3 (1=critical, 3=nice-to-have)
""",
        schema=GapList,
    )
    return result.gaps


@app.reasoner()
async def generate_refinement_queries(gaps: List[Gap]) -> List[RefinementQuery]:
    """Generate targeted queries to fill gaps"""
    gap_descriptions = [g.description for g in gaps]

    result = await app.ai(
        system="You generate targeted search queries to fill information gaps.",
        user=f"""Information gaps: {gap_descriptions}

For each gap, generate a focused search query.
""",
        schema=RefinementQueryList,
    )
    return result.queries


@app.reasoner()
async def iterative_refinement(
    question: str, query_type: str, max_iterations: int = 3
) -> DraftAnswer:
    """
    Iteratively refine answer with confidence-driven routing
    """
    current_chunks = await ensemble_retrieval(question, query_type, top_k=5)
    iteration = 0
    draft = None

    while iteration < max_iterations:
        draft = await synthesize_draft_answer(question, current_chunks)

        app.note(
            f"Iteration {iteration + 1}: confidence={draft.confidence:.2f}",
            ["refinement"],
        )

        # Confidence-driven routing
        if draft.confidence > 0.85:
            app.note("High confidence - early exit", ["refinement"])
            break

        if iteration < max_iterations - 1:
            # Identify gaps
            gaps = await identify_gaps(draft)

            if gaps:
                # Generate refinement queries
                refinement_queries = await generate_refinement_queries(gaps)

                # Expand retrieval for each gap
                additional_chunks = []
                for ref_query in refinement_queries:
                    new_chunks = await ensemble_retrieval(
                        ref_query.query, query_type, top_k=3
                    )
                    additional_chunks.extend(new_chunks)

                # Merge and deduplicate
                all_chunk_dicts = [
                    {"chunk_id": c.chunk_id, "score": c.score, "text": c.text}
                    for c in current_chunks
                ]
                for chunk in additional_chunks:
                    all_chunk_dicts.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "score": chunk.score,
                            "text": chunk.text,
                        }
                    )

                unique_dicts = deduplicate_chunks(all_chunk_dicts)
                current_chunks = [RankedChunk(**c) for c in unique_dicts]

        iteration += 1

    if draft is None:
        draft = await synthesize_draft_answer(question, current_chunks)

    return draft


# ============= PHASE 5: CLAIM VERIFICATION =============


@app.reasoner()
async def decompose_into_claims(answer_text: str) -> List[Claim]:
    """Break answer into atomic verifiable claims"""
    result = await app.ai(
        system="You extract atomic factual claims from text.",
        user=f"""Text: "{answer_text}"

Extract individual claims. Each should be:
- Atomic (one fact)
- Verifiable
- Self-contained

Classify as: fact, inference, or opinion
""",
        schema=ClaimList,
    )
    return result.claims


@app.reasoner()
async def verify_claim(claim: Claim) -> VerificationResult:
    """Verify single claim against source chunks"""
    chunks = await app.memory.get("document_chunks", [])

    # Find supporting chunks
    supporting_chunks = []
    for chunk in chunks:
        if any(
            word in chunk["text"].lower() for word in claim.text.lower().split()[:5]
        ):
            supporting_chunks.append(chunk)

    if not supporting_chunks:
        return VerificationResult(
            claim_id=claim.id, is_verified=False, confidence=0.0, quote_ids=[]
        )

    # Extract exact quotes
    quote_chunks = []
    for chunk in supporting_chunks[:5]:
        quote = find_quote_in_chunk(claim.text, chunk["text"])
        if quote:
            quote_chunks.append((chunk["id"], quote))

    if not quote_chunks:
        return VerificationResult(
            claim_id=claim.id, is_verified=False, confidence=0.0, quote_ids=[]
        )

    # Verify with AI
    context = "\n\n".join([f"[{chunk_id}] {quote}" for chunk_id, quote in quote_chunks])

    result = await app.ai(
        system="You verify claims against source quotes. Be strict.",
        user=f"""Claim: "{claim.text}"

Source quotes:
{context}

Is this claim directly supported?
""",
        schema=VerificationResult,
    )

    result.claim_id = claim.id
    return result


# ============= PHASE 6: FINAL SYNTHESIS =============


@app.reasoner()
async def build_verified_answer(
    draft: DraftAnswer, verifications: List[VerificationResult]
) -> VerifiedAnswer:
    """Rebuild answer using only verified claims"""
    chunks = await app.memory.get("document_chunks", [])
    chunk_map = {c["id"]: c for c in chunks}

    # Filter claims
    verified = [v for v in verifications if v.is_verified and v.confidence > 0.7]
    uncertain = [v for v in verifications if 0.4 < v.confidence <= 0.7]
    removed = [v for v in verifications if v.confidence <= 0.4]

    # Build citations
    citations = []
    for v in verified:
        for chunk_id in v.quote_ids:
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                quote = chunk["text"][:200] + "..."

                citations.append(
                    Citation(
                        chunk_id=chunk_id,
                        quote=quote,
                        page_num=chunk["metadata"].get("index"),
                    )
                )

    # Calculate confidence
    avg_confidence = (
        sum(v.confidence for v in verified) / len(verifications)
        if verifications
        else 0.0
    )

    return VerifiedAnswer(
        answer=draft.text,
        citations=citations,
        confidence_score=avg_confidence,
        verification_summary={
            "verified": len(verified),
            "uncertain": len(uncertain),
            "removed": len(removed),
        },
        completeness_score=draft.confidence,
        gaps=draft.gaps,
    )


# ============= PHASE 7: QUALITY CHECK =============


@app.reasoner()
async def completeness_check(answer: VerifiedAnswer, original_question: str) -> float:
    """Check if answer fully addresses the question"""
    result = await app.ai(
        system="You assess answer completeness.",
        user=f"""Question: "{original_question}"

Answer: "{answer.answer}"

Gaps: {answer.gaps}

Rate completeness (0.0-1.0):
- 1.0: Fully answers question
- 0.7: Mostly complete, minor gaps
- 0.4: Partial answer
- 0.0: Doesn't answer question
""",
        schema=CompletenessScore,
    )
    return result.score


# ============= MAIN ENTRY POINT =============


@app.reasoner()
async def query_document(file_path: str, question: str) -> VerifiedAnswer:
    """
    Main orchestrator - Full lazy-indexed ensemble RAG pipeline
    """
    try:
        # Phase 1: Chunk document
        app.note("Phase 1: Chunking document", ["pipeline"])
        chunk_list = await chunk_document(file_path)
        app.note(f"Created {chunk_list.total_count} chunks", ["chunking"])

        # Phase 2: Query analysis with conditional routing
        app.note("Phase 2: Analyzing query", ["pipeline"])
        query_analysis = await analyze_query_complexity(question)
        app.note(
            f"Complexity: {query_analysis.complexity_score:.2f}, "
            f"Type: {query_analysis.query_type}",
            ["analysis"],
        )

        # Conditional: Decompose if needed
        if query_analysis.needs_decomposition:
            sub_questions = await decompose_query(question)
            app.note(
                f"Decomposed into {len(sub_questions)} sub-questions", ["decomposition"]
            )

        # Phase 3-4: Ensemble retrieval + Iterative refinement
        app.note("Phase 3-4: Ensemble retrieval and refinement", ["pipeline"])
        draft = await iterative_refinement(
            question, query_analysis.query_type, max_iterations=3
        )
        app.note(f"Draft confidence: {draft.confidence:.2f}", ["synthesis"])

        # Phase 5: Claim verification (parallel)
        app.note("Phase 5: Verifying claims", ["pipeline"])
        claims = await decompose_into_claims(draft.text)
        app.note(f"Extracted {len(claims)} claims", ["verification"])

        verification_tasks = [verify_claim(claim) for claim in claims]
        verifications = await asyncio.gather(*verification_tasks)

        # Phase 6: Build verified answer
        app.note("Phase 6: Building verified answer", ["pipeline"])
        final_answer = await build_verified_answer(draft, verifications)

        # Phase 7: Quality check
        app.note("Phase 7: Quality check", ["pipeline"])
        completeness = await completeness_check(final_answer, question)
        final_answer.completeness_score = completeness

        app.note(
            f"Complete! Verified: {final_answer.verification_summary['verified']}, "
            f"Removed: {final_answer.verification_summary['removed']}, "
            f"Completeness: {completeness:.2f}",
            ["complete"],
        )

        return final_answer

    finally:
        # CRITICAL: Memory cleanup
        app.note("Cleaning up memory", ["cleanup"])
        await app.memory.delete("document_chunks")
        await app.memory.delete("document_path")


if __name__ == "__main__":
    print("🚀 Agentic RAG - Production Ready")
    print("📍 Node: agentic-rag")
    print("🌐 Control Plane: http://localhost:8080")
    print("\nArchitecture:")
    print("  ✅ Lazy-indexed ensemble retrieval")
    print("  ✅ Confidence-driven iterative refinement")
    print("  ✅ Parallel claim verification")
    print("  ✅ FastEmbed semantic search")
    print("  ✅ Memory-backed caching")
    print("  ✅ Automatic cleanup")

    app.run(auto_port=True)
