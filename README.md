# Plant-AI Demo —— 真实论文 RAG 链路复刻

从生产项目 `FDM-AI-NEW` 抽取的**真实**检索增强链路：Chroma 向量库、BM25 词法召回、
RRF 融合、qwen3-rerank 重排、证据绑定式作答与数字溯源校验，全部指向本地 75,779 段
真实论文证据。**没有 mock 数据、没有假回答降级**：缺凭证就报错。

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
| 设备手册（脱敏后） | `devices/Device-A/` | 1.9 MB，含 .docx 手册 |
| 设备向量集合 | `plant_Device-A` | 41 段 |

> `data/chroma_db/` 与 `data/papers/` 体积过大，已在 `.gitignore` 中排除；
> 换机器时需从原项目同步这两个目录，然后执行一次集合改名（见文末）。

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

**验证防幻觉**：把同一问题的数字改成原文不存在的值（例如问「干旱下 Fv/Fm 下降 12.5%
是否合理」），校验器会剔除该声明并给出「证据不足」的收敛回答。

## 目录结构

```
app.py                 入口：page_config + 状态初始化 + 侧边栏 + st.navigation
views/
  home.py              首页
  experiment.py        实验方案生成（真实 LLM 流式）
  analysis.py          智能数据分析（真实 CSV 统计 + 图表 + 报告）
  assistant.py         论文问答（真实 RAG）
core/
  config.py            环境变量配置（字段名与原项目一致）
  logging.py           标准库日志适配（替代 loguru）
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
  ...（共 26 个模块，业务逻辑与原项目一致）
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
devices/Device-A/      脱敏后的设备手册与操作规范
```

## 数据说明

企业标识、设备型号与内部路径已统一替换为中性名称（企业 → `Plant Technology Co., Ltd.`，设备 → `Device-A`），
代码、设备文档与向量块三处均已处理。论文语料为公开文献，未做任何内容删改，`叶绿素荧光` 等学科术语保留原样。

## 与原项目的差异

| 原实现 | 现实现 |
|---|---|
| SQLite 会话/历史 + 鉴权 + 加密凭证库 | `st.session_state` + `.demo_state/*.json`，凭证只读 `.env` |
| `loguru` + JSONL 审计 | 标准库 `logging` |
| 加密资源包 `secure_assets.bin` | `devices/` 明文目录（已脱敏） |
| 打包路径探测（exe/one-folder） | 项目根目录相对路径 |
| 手册问答路由 + 联网兜底 + 推荐卡片 | 保留论文链路；手册检索能力（`PlantRetriever`）仍在，未接入页面 |

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
