import {
  startTransition,
  useDeferredValue,
  useEffect,
  useState,
} from "react";
import githubIcon from "../images/github logo.svg";
import openreviewIcon from "../images/openreview.png";
import peachLabLogo from "../images/peach-lab.png";

const DATASETS = [
  { key: "ShareGPT", label: "ShareGPT", tone: "Open-domain assistant prompts" },
  { key: "wildchat", label: "WildChat", tone: "Wild conversational prompts" },
  { key: "medQA", label: "MedQA", tone: "Clinical multiple-choice questions" },
  { key: "casehold", label: "CaseHOLD", tone: "Legal multiple-choice holdings" },
];

const MODEL_LABELS = {
  "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
  "claude-sonnet-4-20250514": "Claude Sonnet 4",
  "gpt-4.1-nano": "GPT-4.1 nano",
  "gpt-4.1": "GPT-4.1",
  "gpt-5": "GPT-5",
  "lgai_exaone-deep-32b": "EXAONE Deep 32B",
  "mistralai_mistral-small-3.1-24b-instruct": "Mistral Small 3.1",
  "qwen_qwen-2.5-7b-instruct": "Qwen 2.5 7B Instruct",
  "local_qwen2.5-0.5b-instruct": "Qwen 2.5 0.5B Instruct",
};

const ACTION_ORDER = ["redact", "abstract", "retain"];
const PANE_MODES = [
  { key: "both", label: "Both" },
  { key: "oracle", label: "Oracle only" },
  { key: "prediction", label: "Prediction only" },
];

const PAPER = {
  title: "Operationalizing Data Minimization for Privacy-Preserving LLM Prompting",
  subtitle:
    "Project website and interactive explorer for oracle versus model-predicted minimization across four datasets and nine response models.",
  authors: "Jijie Zhou, Niloofar Mireshghallah, Tianshi Li",
  openreviewUrl: "https://openreview.net/forum?id=rpcnvW33EG",
  abstract:
    "The rapid deployment of large language models (LLMs) in consumer applications has led to frequent exchanges of personal information. To obtain useful responses, users often share more than necessary, increasing privacy risks via memorization, context-based personalization, or security breaches. We present a framework to formally define and operationalize data minimization: for a given user prompt and response model, quantifying the least privacy-revealing disclosure that maintains utility, and propose a priority-queue tree search to locate this optimal point within a privacy-ordered transformation space. We evaluated the framework on four datasets spanning open-ended conversations (ShareGPT, WildChat) and knowledge-intensive tasks with single-ground-truth answers (CaseHOLD, MedQA), quantifying achievable data minimization with nine LLMs as the response model. Our results demonstrate that larger frontier LLMs can tolerate stronger data minimization while maintaining task quality than smaller open-source models (85.7% redaction for GPT-5 vs. 19.3% for Qwen2.5-0.5B). By comparing with our search-derived benchmarks, we find that LLMs struggle to predict optimal data minimization directly, showing a bias toward abstraction that leads to oversharing. This suggests not just a privacy gap, but a capability gap: models may lack awareness of what information they actually need to solve a task.",
  codeUrl: "https://github.com/PEACH-Research-Lab/Operationalize-Data-Minimization/",
  bibtex: `@inproceedings{
zhou2026operationalizing,
title={Operationalizing Data Minimization for Privacy-Preserving {LLM} Prompting},
author={Jijie Zhou and Niloofar Mireshghallah and Tianshi Li},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=rpcnvW33EG}
}`,
};

function App() {
  const [datasetKey, setDatasetKey] = useState(DATASETS[0].key);
  const [datasetCache, setDatasetCache] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedModel, setSelectedModel] = useState("gpt-4.1");
  const [selectedRecordId, setSelectedRecordId] = useState("");
  const [paneMode, setPaneMode] = useState("both");
  const [search, setSearch] = useState("");

  const deferredSearch = useDeferredValue(search);

  useEffect(() => {
    let cancelled = false;

    async function loadDataset() {
      if (datasetCache[datasetKey]) {
        return;
      }

      setLoading(true);
      setError("");
      try {
        const response = await fetch(
          `${import.meta.env.BASE_URL}data/${datasetKey}.json`,
        );
        if (!response.ok) {
          throw new Error(`Failed to load ${datasetKey}.json`);
        }
        const payload = await response.json();
        if (cancelled) {
          return;
        }
        setDatasetCache((current) => ({ ...current, [datasetKey]: payload }));
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load data.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadDataset();

    return () => {
      cancelled = true;
    };
  }, [datasetCache, datasetKey]);

  const dataset = datasetCache[datasetKey];
  const models = dataset?.models ?? [];

  useEffect(() => {
    if (!models.length) {
      return;
    }
    if (!models.includes(selectedModel)) {
      setSelectedModel(models[0]);
    }
  }, [models, selectedModel]);

  const filteredRecords = (() => {
    if (!dataset?.records) {
      return [];
    }

    const query = deferredSearch.trim().toLowerCase();
    if (!query) {
      return dataset.records;
    }

    return dataset.records.filter((record) => {
      const searchCorpus = [
        record.index,
        record.oracle_index_value,
        record.prediction_index_value,
        record.original?.text,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return searchCorpus.includes(query);
    });
  })();

  useEffect(() => {
    if (!filteredRecords.length) {
      setSelectedRecordId("");
      return;
    }
    const existing = filteredRecords.some((record) => record.record_id === selectedRecordId);
    if (!existing) {
      setSelectedRecordId(filteredRecords[0].record_id);
    }
  }, [filteredRecords, selectedRecordId]);

  const selectedRecord =
    filteredRecords.find((record) => record.record_id === selectedRecordId) ??
    filteredRecords[0] ??
    null;

  const selectedIndex = filteredRecords.findIndex(
    (record) => record.record_id === selectedRecord?.record_id,
  );

  const selectedResult = selectedRecord?.models?.[selectedModel] ?? null;
  const actionRows = selectedResult
    ? buildActionRows(selectedResult.oracle, selectedResult.prediction)
    : [];

  const summary = buildSummary(actionRows);

  function handleDatasetChange(nextDataset) {
    startTransition(() => {
      setDatasetKey(nextDataset);
      setSearch("");
      setSelectedRecordId("");
    });
  }

  function handleMove(delta) {
    if (selectedIndex < 0) {
      return;
    }
    const nextIndex = Math.min(
      Math.max(selectedIndex + delta, 0),
      filteredRecords.length - 1,
    );
    setSelectedRecordId(filteredRecords[nextIndex].record_id);
  }

  function handleRandom() {
    if (!filteredRecords.length) {
      return;
    }
    const randomRecord =
      filteredRecords[Math.floor(Math.random() * filteredRecords.length)];
    setSelectedRecordId(randomRecord.record_id);
  }

  return (
    <div className="site-shell">
      <header className="hero">
        <div className="hero__inner">
          <img
            className="hero__logo"
            src={peachLabLogo}
            alt="PEACH Lab logo"
          />
          <div className="eyebrow">ICLR Project Website</div>
          <div className="hero__title-row">
            <h1>{PAPER.title}</h1>
          </div>
          <p className="hero__subtitle">{PAPER.subtitle}</p>
          <p className="hero__authors">{PAPER.authors}</p>
          <div className="hero__actions">
            <a className="button button--primary" href="#explorer">
              Open Explorer
            </a>
            <a className="button button--ghost" href="#bibtex">
              BibTeX
            </a>
            <a
              className="icon-link"
              href={PAPER.codeUrl}
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub repository"
              title="GitHub repository"
            >
              <img src={githubIcon} alt="" />
            </a>
            <a
              className="icon-link"
              href={PAPER.openreviewUrl}
              target="_blank"
              rel="noreferrer"
              aria-label="OpenReview page"
              title="OpenReview page"
            >
              <img src={openreviewIcon} alt="" />
            </a>
          </div>
        </div>
      </header>

      <main className="page">
        <section className="paper-grid">
          <article className="paper-card paper-card--wide">
            <span className="section-label">Abstract</span>
            <p>{PAPER.abstract}</p>
          </article>
          <article className="paper-card">
            <span className="section-label">Code</span>
            <p>
              Code, data processing scripts, supporting resources, and the
              OpenReview page for this project are linked below.
            </p>
            <a className="paper-link" href={PAPER.codeUrl} target="_blank" rel="noreferrer">
              {PAPER.codeUrl}
            </a>
            <a
              className="paper-link"
              href={PAPER.openreviewUrl}
              target="_blank"
              rel="noreferrer"
            >
              {PAPER.openreviewUrl}
            </a>
          </article>
        </section>

        <section className="dataset-strip">
          {DATASETS.map((datasetMeta) => (
            <button
              key={datasetMeta.key}
              type="button"
              className={`dataset-pill ${
                datasetMeta.key === datasetKey ? "dataset-pill--active" : ""
              }`}
              onClick={() => handleDatasetChange(datasetMeta.key)}
            >
              <span>{datasetMeta.label}</span>
              <small>{datasetMeta.tone}</small>
            </button>
          ))}
        </section>

        <section id="explorer" className="explorer">
          <div className="explorer__header">
            <div>
              <span className="section-label">Interactive Explorer</span>
              <h2>Inspect oracle and predicted minimization at the message level</h2>
              <p>
                Select a dataset, target model, and message to compare how oracle
                minimization and model-predicted minimization diverge.
              </p>
            </div>
            {selectedRecord ? (
              <div className="stats-row">
                <StatCard label="Records" value={dataset?.record_count ?? 0} />
                <StatCard label="Matched spans" value={summary.matches} />
                <StatCard label="Mismatched spans" value={summary.mismatches} />
              </div>
            ) : null}
          </div>

          <div className="controls">
            <div className="control">
              <label htmlFor="model-select">Model</label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={!models.length}
              >
                {models.map((model) => (
                  <option key={model} value={model}>
                    {MODEL_LABELS[model] ?? model}
                  </option>
                ))}
              </select>
            </div>

            <div className="control control--wide">
              <label htmlFor="message-search">Search messages</label>
              <input
                id="message-search"
                type="text"
                placeholder="Search by index, id, or prompt text"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </div>

            <div className="control control--wide">
              <label htmlFor="message-select">Message</label>
              <select
                id="message-select"
                value={selectedRecord?.record_id ?? ""}
                onChange={(event) => setSelectedRecordId(event.target.value)}
                disabled={!filteredRecords.length}
              >
                {filteredRecords.map((record) => (
                  <option key={record.record_id} value={record.record_id}>
                    {formatRecordOption(record)}
                  </option>
                ))}
              </select>
            </div>

            <div className="control">
              <label>Browse</label>
              <div className="inline-actions">
                <button
                  type="button"
                  className="button button--ghost"
                  onClick={() => handleMove(-1)}
                  disabled={selectedIndex <= 0}
                >
                  Prev
                </button>
                <button
                  type="button"
                  className="button button--ghost"
                  onClick={handleRandom}
                  disabled={!filteredRecords.length}
                >
                  Random
                </button>
                <button
                  type="button"
                  className="button button--ghost"
                  onClick={() => handleMove(1)}
                  disabled={
                    selectedIndex < 0 || selectedIndex >= filteredRecords.length - 1
                  }
                >
                  Next
                </button>
              </div>
            </div>
          </div>

          <div className="pane-toggle" role="tablist" aria-label="Explorer views">
            {PANE_MODES.map((mode) => (
              <button
                key={mode.key}
                type="button"
                className={`pane-toggle__button ${
                  paneMode === mode.key ? "pane-toggle__button--active" : ""
                }`}
                onClick={() => setPaneMode(mode.key)}
              >
                {mode.label}
              </button>
            ))}
          </div>

          {loading && !dataset ? <div className="notice">Loading dataset...</div> : null}
          {error ? <div className="notice notice--error">{error}</div> : null}

          {selectedRecord && selectedResult ? (
            <>
              <div className="record-meta">
                <MetaPill label={selectedRecord.oracle_index_key} value={selectedRecord.oracle_index_value} />
                <MetaPill
                  label={selectedRecord.prediction_index_key}
                  value={selectedRecord.prediction_index_value}
                />
                <MetaPill label="index" value={selectedRecord.index} />
                <MetaPill
                  label="model"
                  value={MODEL_LABELS[selectedModel] ?? selectedModel}
                />
              </div>

              <div className="viewer-grid">
                <section className="viewer-card viewer-card--original">
                  <div className="viewer-card__header">
                    <span className="section-label">Original Input</span>
                    <h3>Prompt</h3>
                  </div>
                  <PromptBlock original={selectedRecord.original} />
                </section>

                <section className="viewer-card viewer-card--result">
                  <div className="viewer-card__header">
                    <span className="section-label">Minimization Output</span>
                    <h3>Oracle and prediction</h3>
                  </div>

                  {paneMode !== "prediction" ? (
                    <ResultPanel
                      title="Oracle"
                      subtitle="Precomputed oracle minimization"
                      result={selectedResult.oracle}
                    />
                  ) : null}

                  {paneMode === "both" ? <div className="divider" /> : null}

                  {paneMode !== "oracle" ? (
                    <ResultPanel
                      title="Prediction"
                      subtitle="Model-predicted minimization"
                      result={selectedResult.prediction}
                      unavailableReason={selectedResult.prediction_unavailable_reason}
                    />
                  ) : null}
                </section>
              </div>

              <section className="table-card">
                <div className="table-card__header">
                  <div>
                    <span className="section-label">Span Actions</span>
                    <h3>Action-level breakdown</h3>
                  </div>
                  <div className="summary-row">
                    <SummaryPill label="Match" value={summary.matches} tone="match" />
                    <SummaryPill
                      label="Mismatch"
                      value={summary.mismatches}
                      tone="mismatch"
                    />
                  </div>
                </div>

                {actionRows.length ? (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Span</th>
                          <th>Type</th>
                          <th>Oracle</th>
                          <th>Prediction</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {actionRows.map((row) => (
                          <tr key={row.span}>
                            <td className="span-cell">
                              <strong>{row.span}</strong>
                              <small>{row.replacementPreview}</small>
                            </td>
                            <td>{row.type ?? "Unknown"}</td>
                            <td>
                              <ActionTag action={row.oracleAction} />
                            </td>
                            <td>
                              {row.predictionAction ? (
                                <ActionTag action={row.predictionAction} />
                              ) : (
                                <span className="muted">Unavailable</span>
                              )}
                            </td>
                            <td>
                              <span
                                className={`status-badge ${
                                  row.matches ? "status-badge--match" : "status-badge--mismatch"
                                }`}
                              >
                                {row.matches ? "Match" : "Mismatch"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="empty-state">No sensitive spans were detected for this record.</div>
                )}
              </section>
            </>
          ) : (
            !loading && <div className="empty-state">No record matches the current filters.</div>
          )}
        </section>

        <section id="bibtex" className="paper-card paper-card--wide">
          <span className="section-label">BibTeX</span>
          <pre className="bibtex-block">{PAPER.bibtex}</pre>
        </section>
      </main>
    </div>
  );
}

function PromptBlock({ original }) {
  return (
    <div className="prompt-block">
      <p>{original.text}</p>
      {Array.isArray(original.choices) && original.choices.length ? (
        <div className="choices">
          <h4>Choices</h4>
          <ol>
            {original.choices.map((choice, index) => (
              <li key={`${index}-${choice}`}>{choice}</li>
            ))}
          </ol>
          <div className="correct-choice">
            Correct choice: <strong>{original.correct_choice}</strong>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ResultPanel({ title, subtitle, result, unavailableReason }) {
  return (
    <div className="result-panel">
      <div className="result-panel__header">
        <div>
          <h4>{title}</h4>
          <p>{subtitle}</p>
        </div>
        {result?.transformation_stats?.totals ? (
          <div className="mini-stats">
            {ACTION_ORDER.map((action) => (
              <SummaryPill
                key={action}
                label={capitalize(action)}
                value={result.transformation_stats.totals[action] ?? 0}
                tone={action}
              />
            ))}
          </div>
        ) : null}
      </div>

      {result ? (
        <>
          <div className="masked-box">
            <span className="section-label">Masked Text</span>
            <p>{result.masked_text || "No masked text available."}</p>
          </div>

          {result.explanation ? (
            <div className="explanation-box">
              <span className="section-label">Rationale</span>
              <p>{result.explanation}</p>
            </div>
          ) : null}
        </>
      ) : (
        <div className="notice">{unavailableReason || "No prediction output available."}</div>
      )}
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetaPill({ label, value }) {
  return (
    <div className="meta-pill">
      <span>{label}</span>
      <strong>{String(value)}</strong>
    </div>
  );
}

function SummaryPill({ label, value, tone }) {
  return (
    <div className={`summary-pill summary-pill--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ActionTag({ action }) {
  return <span className={`action-tag action-tag--${action}`}>{capitalize(action)}</span>;
}

function buildActionRows(oracle, prediction) {
  const piiDict = oracle?.pii_dict || prediction?.pii_dict || {};
  const spans = Object.keys(piiDict);

  return spans
    .map((span) => {
      const oracleAction = oracle?.transformation?.[span] || null;
      const predictionAction = prediction?.transformation?.[span] || null;
      return {
        span,
        type: piiDict[span],
        oracleAction,
        predictionAction,
        matches: oracleAction === predictionAction,
        replacementPreview: formatReplacementPreview(span, oracle, prediction, oracleAction),
      };
    })
    .sort((left, right) => {
      if (left.matches === right.matches) {
        return left.span.localeCompare(right.span);
      }
      return left.matches ? 1 : -1;
    });
}

function buildSummary(rows) {
  const matches = rows.filter((row) => row.matches).length;
  return {
    matches,
    mismatches: rows.length - matches,
  };
}

function formatReplacementPreview(span, oracle, prediction, oracleAction) {
  const redactValue =
    oracle?.redact_map?.[span] ??
    prediction?.redact_map?.[span] ??
    oracle?.replacement_map?.[span] ??
    prediction?.replacement_map?.[span];

  const abstractValue =
    oracle?.abstract_map?.[span] ??
    prediction?.abstract_map?.[span] ??
    oracle?.replacement_map?.[span] ??
    prediction?.replacement_map?.[span];

  if (oracleAction === "redact" && redactValue) {
    return `oracle output: ${redactValue}`;
  }
  if (oracleAction === "abstract" && abstractValue) {
    return `oracle output: ${abstractValue}`;
  }
  if (oracleAction === "retain") {
    return "oracle output: retained";
  }
  return "oracle output: none";
}

function formatRecordOption(record) {
  const preview = record.original?.text?.replace(/\s+/g, " ").trim() ?? "";
  const truncated = preview.length > 90 ? `${preview.slice(0, 87)}...` : preview;
  return `[${record.index}] ${truncated}`;
}

function capitalize(value) {
  if (!value) {
    return "";
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export default App;
