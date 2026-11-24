# Plant-AI Demo —— 可溯源的论文 RAG 问答

一套完整的检索增强问答链路：Chroma 向量库、BM25 词法召回、RRF 融合、
qwen3-rerank 重排、证据绑定式作答与数字溯源校验，指向本地 75,779 段真实
论文证据。**没有 mock 数据、没有假回答兜底**：缺凭证就报错，检索不到就拒答。

## 链路总览

```
用户问题
   │
   ├─ QueryPlan 解析（实体约束 / 对比意图 / 必需证据槽位）
   │
   ├─ Chroma 向量召回  ──┐        text-embedding-v4（1024 维，75,779 chunks）
   │                     ├─ RRF 融合（k=30，实体约束感知重排）
   ├─ BM25 词法召回  ───┘        bm25_corpus.jsonl（104MB）
   │
   ├─ qwen3-rerank 二次重排（Top-N 收敛，锁定主论文）
   │
   ├─ 收敛式作答（DeepSeek）：强制声明式输出 + evidence_id
   │
   ├─ citation_validator：逐条核对数字是否出现在原文片段
   │     └─ 无出处的数字整条剔除，无证据则拒答
   │
   └─ 参考文献渲染：真实书目 + https://doi.org/… 可跳转链接
```

## 数据资产（已在本地）

| 资产 | 路径 | 规模 |
|---|---|---|
| Chroma 向量库 | `data/chroma_db` | 2.1 GB，含 8 个集合 |
| 活跃论文集合 | `plant_papers_pilot_full1979_v1` | **75,779** 段证据 |
| BM25 语料 | `data/papers/.../retrieval_assets/bm25_corpus.jsonl` | 104 MB |
| 论文实体索引 | `.../paper_entities.jsonl` | 6,292 KB |
| 实体别名目录 | `.../entity_aliases.jsonl` | 23 KB |
| 参考文献目录 | `.../selection/proposed_all_fulltext.jsonl` | 5.6 MB |
| 研究类型标签 | `.../paper_research_tags.jsonl` | 793 KB |
| 实验事实卡 | `.../fact_cards/fact_cards.jsonl` | 905 KB |
| 补充语料（数据分析向） | `data/papers/data_analysis_evidence_index/` | 51 MB |
| 设备知识文档 | `devices/Device-A/` | 1.9 MB，含 .docx 手册与操作规范 |
| 设备向量集合 | `plant_Device-A` | 41 段 |

> `data/chroma_db/` 与 `data/papers/` 体积过大，已在 `.gitignore` 中排除，
> 仓库只包含代码与设备文档。因此**克隆后没有数据是跑不通检索的**，
> 下文「如何验证链路是真的」给了不依赖数据的核验方式。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env      # 填入 LLM_API_KEY（DeepSeek）与 EMBEDDING_API_KEY（DashScope）
streamlit run app.py      # http://localhost:8501
```

两把 Key 缺一不可：

- `LLM_API_KEY`：DeepSeek，负责作答与叙述；缺失则页面直接提示「请配置 Key」
- `EMBEDDING_API_KEY`：阿里云 DashScope，`text-embedding-v4` 负责查询向量化；
  缺失则无法召回（也报同样的错）

## 如何验证「真的在检索」

**命令行（最快）**

```bash
python example_run.py
```

预期输出（约 20 秒）：

```
干旱胁迫下Fv/Fm显著下降（Fig. 2B）。
与对照相比，Fv/Fm平均值下降6.8%（从0.74±0.01降至0.69±0.13）。
...
参考文献：
[1] Zhuang J, Wang Y, ... Drought stress strengthens the link between chlorophyll
fluorescence parameters and photosynthetic traits [J]. 2020.
https://doi.org/10.7717/peerj.10046. doi: 10.7717/peerj.10046.
```

**网页端**：进入「科研问答助手」，输入

> 干旱胁迫下叶绿素荧光参数 Fv/Fm 的变化及其生理意义是什么？

应看到三段可核验的信息：

1. 状态条显示 `完成：命中 10 段证据`
2. 「参考来源」折叠区里有真实书目与 `https://doi.org/…` 链接（点开即跳出版社）
3. 「检索证据（10 段）」折叠区列出每段的重排得分（0.75~0.90）与章节路径，
   说明证据来自 Chroma 召回结果而非模型臆造

**验证防幻觉**（两层，拦在不同位置）：

1. **论文锁**：Top1 重排分与 Top2 的差值小于 `0.10` 时，判定"命中对象不唯一"，
   中断生成并列出候选论文让用户澄清（`rag/scalable_retrieval.py` 的
   `assess_paper_lock_confidence`）。此时大模型尚未被调用，任何自然语言指令都
   无法绕过这一层。
2. **数字溯源**：即使锁定了论文，模型输出的每个数值仍要与证据切片比对，
   原文里找不到的数字整条剔除（`rag/citation_validator.py`）。这一层可以直接
   单测：构造一条证据 + 两条声明（`6.8%` 与虚构的 `12.5%`），调用
   `validate_and_render_answer(..., allow_partial_claims=True)`，结果应为
   `accepted: 1, dropped: 1, warnings: ('claim_2_unsupported_numbers:12.5%',)`。

## 目录结构

```
app.py                 入口：page_config + 状态初始化 + 侧边栏 + st.navigation
views/
  home.py              首页
  experiment.py        实验方案生成（真实 LLM 流式）
  analysis.py          智能数据分析（真实 CSV 统计 + 图表 + 报告）
  assistant.py         论文问答（真实 RAG）
core/
  config.py            环境变量配置（env 驱动，字段集中定义）
  logging.py           标准库日志封装（统一格式）
rag/
  vector_store.py      Chroma + DashScope Embedding
  chroma_paper_search.py  向量召回
  local_chunk_search.py   BM25 词法召回
  paper_rank_fusion.py    RRF 融合 + 实体约束重排
  scalable_retrieval.py   混合召回编排、论文锁定
  provider_adapters.py    Qwen 重排 / 覆盖式作答适配
  citation_validator.py   数字溯源校验
  paper_citations.py      书目与 DOI 渲染
  retriever.py            设备手册混合检索（Device-A）
  document_loader.py      设备 .docx/.pdf 分段
  ...（共 26 个模块，按召回 / 融合 / 校验 / 渲染分层）
evaluation/
  qwen_rerank.py       DashScope text-rerank 客户端（SDK 缺失时走 HTTP）
  local_hybrid.py      BM25 索引
agents/
  analysis_agent.py    LangGraph 四节点工作流
  llm_factory.py       ChatOpenAI 工厂
services/
  fulltext_paper_rag_service.py  论文问答门面
  paper_rag_factory.py / paper_rag_pipeline.py  装配与流水线
  llm_gateway.py       凭证检查与客户端构建
  export.py            Markdown → docx
devices/Device-A/      设备手册与操作规范
```

## 数据说明

论文语料为公开文献，未做内容删改。设备文档与向量分块中的企业标识、设备型号、
内部路径统一为中性名称（`Plant Technology Co., Ltd.` / `Device-A`），
`叶绿素荧光` 等学科术语保留原样。

## 工程取舍

| 决策 | 做法 | 理由 |
|---|---|---|
| 会话存储 | `st.session_state` + `.demo_state/*.json` | 单机使用不需要数据库，刷新后仍可恢复 |
| 凭证管理 | 只读 `.env`，缺失即报错 | 密钥不落盘，也不做"看起来能用"的降级 |
| 设备资源 | `devices/` 明文目录 | clone 后可直接阅读、直接进 RAG 索引 |
| 路径解析 | 项目根目录相对路径 | 不依赖打包形态，迁移成本低 |
| 手册问答 | `PlantRetriever` 可用，未接入页面 | 本仓库聚焦论文链路，手册检索留作扩展点 |

## 换机器后如何对齐集合名

集合名以 `PAPER_RAG_ACTIVE_COLLECTION_NAME` 为准。若同步来的库仍带旧前缀，用下面脚本统一改名：

```python
import chromadb

OLD_PREFIX = "old_"
client = chromadb.PersistentClient(path="data/chroma_db")
for col in client.list_collections():
    if col.name.startswith(OLD_PREFIX):
        client.get_collection(col.name).modify(name=col.name.replace(OLD_PREFIX, "plant_", 1))
```
