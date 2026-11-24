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

## 三个页面的操作步骤

三个页面都从左侧导航进入。下面用到的示例问题与示例数据都可以直接复制使用。

### 一、科研问答助手（论文 RAG）

**① 确认凭证**

侧边栏底部应显示绿色 `✅ 已通过本地配置（.env）连接大模型` 和一串打码的 Key。
如果显示的是输入框，说明 `.env` 没被读到，先回「快速开始」检查两把 Key。

**② 提问**

点左侧「科研问答助手」，把下面这句粘进输入框：

> 干旱胁迫下叶绿素荧光参数 Fv/Fm 的变化及其生理意义是什么？

输入框下方有三个推荐问题按钮（第一个就是上面这句），点一下等同于发送：

> 水分胁迫下 PSII 光化学效率 Y(II) 如何响应？
>
> 干旱条件下 NPQ 与 Y(NO) 的变化趋势有何差异？

单次回答约 15–25 秒，这期间检索、重排、作答是串行真实调用，不要刷新页面。

**③ 看四处信息**

状态条会依次走过四个阶段：

1. 解析问题意图与必需证据槽位
2. Chroma 向量召回 + BM25 词法召回 → RRF 融合
3. qwen3-rerank 二次重排，收敛逐篇证据
4. 校验数字与证据 ID，渲染参考文献

结束时变成 `完成：命中 10 段证据`。然后依次核对：

| 位置 | 应该看到 |
|---|---|
| 正文 | 类似 `下降 6.8%（从 0.74±0.01 降至 0.69±0.13）` 的句子，数字与证据切片一致 |
| 「参考来源」折叠区 | 真实书目 + `https://doi.org/10.7717/peerj.10046` 可点链接 |
| 「检索证据（10 段）」折叠区 | 每段列出重排得分，从 0.89 递减到 0.52 左右，附章节路径 |

得分单调递减，说明排序确实被重排模型重算过，而不是原样返回向量相似度。

**④ 追问**

接着问「那复水之后能恢复吗？」，系统会在同一篇主论文内继续检索，不需要重述背景。
左侧列表可切换历史对话，`🗑` 删除，`清空当前对话` 只清当前这条。

**⑤ 故意触发论文锁（可选）**

把问题换成：

> 干旱下 Fv/Fm 下降 12.5% 这个结论出自哪篇论文？

`12.5%` 在原文里不存在，问题又没锁定物种，于是 Top1 与 Top2 的重排分差小于 `0.10`，
系统判定"命中对象不唯一"，中断生成并列出候选论文（橄榄、柳杉等）要求澄清。

**这不是报错**。此时大模型还没被调用，属于第一层护栏：宁可不答，也不把 A 论文的数值安到 B 论文上。

### 二、智能数据分析

**① 用哪份数据**

仓库自带两份样例，在 `examples/` 目录：

| 文件 | 行数 | 数据形态 | 会触发的图表 |
|---|---|---|---|
| `examples/demo_fluorescence_aq.csv` | 85 | 7 档 PAR × 2 个品种 × 6 重复 | 光响应曲线 + 两组均值比较柱状图 |
| `examples/demo_fluorescence_kinetics.csv` | 25 | 单光强、带 `Cycle=Light_1…` | 荧光动力学时间序列曲线 |

**② 上传与填参**

点左侧「智能数据分析」，把 `examples/demo_fluorescence_aq.csv` 拖进上传区（支持多文件合并）。

下面两项**影响报告措辞、不参与计算**：

- 研究对象/物种（必填）：`水稻`
- 处理与分组含义（必填）：`plate1 为 25℃ 对照，plate2 为 4℃ 冷胁迫 7 天`

可选项可以留空：

- 处理或测量时间：`处理第 0、3、5 天`
- 数据标签说明：`plate1_1–plate1_6 表示对照组的 6 个独立单株`

配色方案随便选一个。

**③ 两段式确认**

点「开始智能分析」，先弹「分析预览」对话框，显示：

- 将生成的图表类型（多组均值比较柱状图 / 光响应曲线）
- 品种数 `2`、PAR 梯度 `7`、是否有重复结构

确认无误再点「确认并开始分析」。这一步是为了避免白跑一次 1–2 分钟的模型调用。

预览里可能出现一句提示：柱状图固定在高光强 `PAR=419.58` 的切片上比较。
这是业务约束——不同光强下的 Fv/Fm 不能混在一起求总平均，代码会自动取覆盖样品最多的那一档。

**④ 结果与导出**

报告分段渲染：文字走 Markdown，图片用 `st.image` 内联在对应位置。底部三个导出入口：

- `下载报告 (.docx)`
- 每张图一个 `图表N - 类型 (.png)`
- 每张图一个 `代码N - 参数 (.py)`

最后一个值得点开看一眼：它是独立可跑的复现脚本，数据路径、参数、绘图代码都写死在里面，
脱离这套系统也能重跑出同一张图。

**⑤ 换一份数据做对照**

再上传 `examples/demo_fluorescence_kinetics.csv` 跑一次。预览里图表类型会变成
「荧光动力学时间序列曲线」，PAR 梯度变成 `1`，报告结论也换一套说法。

同一个系统、两份数据、两条不同路径——这能直观看出图表类型是**代码规则链**
（按列名、PAR 唯一值个数、是否有 Cycle 列逐级判断）算出来的，不是模型猜的。

**⑥ 数据形态不对会怎样**

样例 CSV 里每个重复样品各有一个时间戳，会被判成"同一株测了 6 个时间点"，
于是多出几张几乎水平的动力学曲线；同时 6 个重复数值完全相同，组内标准差是 `0.0`，
Welch t 检验的 p 值没有统计意义。

这两处报告自己会写出来——"与生物学重复的常规离散度不符，建议核查导出流程是否对重复样品作了均值汇总"。
**模型只在程序算出来的数上解释，数据不够就明说不够**，这也是这一页的设计意图。

### 三、实验方案生成

**① 输入需求**

点左侧「实验方案生成」，在左栏输入框粘一段需求，比如：

> 需要对两组番茄幼苗做 4℃ 冷胁迫处理，分别在 0 天、3 天、5 天测量暗适应 Fv/Fm 与光响应曲线，比较两组耐冷性差异，请给出完整测量方案与参数设置。

换成你自己的对象、处理温度、时间点和指标即可，写得越具体，方案越贴合。

**② 生成**

点「生成实验方案」，右侧会一个字一个字流出来（20–40 秒），不是等全部生成完再一次性显示。

**③ 检查结构**

输出应严格按六个章节排列，顺序不乱：

1. 实验目的
2. 材料与样品准备
3. 仪器与参数设置
4. 操作步骤
5. 数据记录与质控
6. 注意事项

末尾是「参考来源」折叠区，默认收起，点开才展开。

**④ 检查"不编参数"**

在正文里搜「以设备手册为准」。正常情况下，3.2 的参数表里几乎每行都写着"以设备手册推荐档为准"，
注意事项里也会有一条"本方案不指定具体档位"。

这是提示词里的硬约束：档位不确定就承认不确定，不许编出"饱和脉冲设 8000 μmol"这类具体数字。
搜到这行字，说明约束生效了。

**⑤ 导出**

左下「下载 Word 文档」，得到排版好的 `.docx`，标题层级与表格都会保留。

> **一个必须知道的差异**：这一页是**纯提示词生成，不走向量检索**。
> 参考来源是模型按自身知识写的（泛指设备手册、通用规范），没有作者、年份、DOI，**不可溯源**；
> 能溯源的是问答页那条链路。被抓着问"你这页的引用能查证吗"，就照实说，别把两件事混着讲。

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
examples/              两份样例 CSV（光响应型 / 动力学型），可直接用于数据分析页
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
