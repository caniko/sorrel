-- Rebuild-only projection. JSON/TSV are authoritative. JSON payloads stay TEXT
-- (PostgreSQL may use jsonb; use BIGINT for bytes, counts and token totals,
-- and DOUBLE PRECISION to match SQLite REAL). Replace the SQLite GLOB
-- guards with PostgreSQL ~ '^[0-9a-f]{64}$' and ~ '^[0-9]{6}$', respectively.
CREATE TABLE dandisets (
    dandiset_id TEXT PRIMARY KEY NOT NULL CHECK (
        length(dandiset_id) = 6 AND dandiset_id NOT GLOB '*[^0-9]*'
    ),
    name TEXT,
    doi TEXT,
    license TEXT,
    description TEXT,
    citation TEXT,
    contributors_json TEXT,
    measurement_techniques_json TEXT,
    anatomy_json TEXT,
    keywords_json TEXT,
    dandi_metadata_json TEXT,
    unique_metadata_json TEXT
);

-- Archive identity, independent of a particular version/snapshot. The observed
-- path, checksum and size belong to review_assets, not this identity table.
CREATE TABLE assets (
    dandiset_id TEXT NOT NULL REFERENCES dandisets(dandiset_id),
    asset_id TEXT NOT NULL,
    PRIMARY KEY (dandiset_id, asset_id)
);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY NOT NULL CHECK (
        length(review_id) = 64 AND review_id NOT GLOB '*[^0-9a-f]*'
    ),
    dandiset_id TEXT NOT NULL REFERENCES dandisets(dandiset_id),
    version TEXT NOT NULL,
    draft_modified TEXT,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('human', 'agent', 'hybrid')),
    contributor TEXT NOT NULL,
    sorrel_version TEXT NOT NULL,
    sorrel_rev TEXT NOT NULL,
    skill_rev TEXT NOT NULL,
    prompt_rev TEXT,
    learned_rev TEXT,
    selection_policy TEXT NOT NULL CHECK (selection_policy IN ('all', 'sample')),
    selection_scope TEXT NOT NULL CHECK (selection_scope IN ('archive', 'nwb')),
    n_assets_total INTEGER NOT NULL CHECK (n_assets_total >= 1),
    n_nwb_assets INTEGER CHECK (n_nwb_assets BETWEEN 0 AND n_assets_total),
    wall_s REAL CHECK (wall_s >= 0),
    cost_usd REAL CHECK (cost_usd >= 0),
    supersedes TEXT,
    review_json TEXT NOT NULL,
    unique_metadata_json TEXT,
    UNIQUE (review_id, dandiset_id),
    FOREIGN KEY (supersedes, dandiset_id) REFERENCES reviews(review_id, dandiset_id)
        DEFERRABLE INITIALLY DEFERRED,
    CHECK (supersedes IS NULL OR supersedes <> review_id),
    CHECK (version <> 'draft' OR draft_modified IS NOT NULL)
);

CREATE TABLE review_models (
    review_id TEXT NOT NULL REFERENCES reviews(review_id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    role TEXT NOT NULL CHECK (role IN ('curation', 'report', 'verification')),
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    harness TEXT,
    effort TEXT,
    turns INTEGER CHECK (turns >= 0),
    input_tokens INTEGER CHECK (input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens >= 0),
    cost_usd REAL CHECK (cost_usd >= 0),
    usage_json TEXT,
    PRIMARY KEY (review_id, ordinal)
);

CREATE TABLE review_assets (
    review_id TEXT NOT NULL,
    dandiset_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    encoding TEXT NOT NULL CHECK (encoding IN ('hdf5', 'zarr', 'other')),
    is_nwb INTEGER NOT NULL CHECK (is_nwb IN (0, 1)),
    open_status TEXT NOT NULL CHECK (open_status IN ('ok', 'refused', 'failed')),
    sha256 TEXT CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    zarr_checksum TEXT,
    locator TEXT,
    media_type TEXT,
    download_url TEXT,
    blob_id TEXT,
    zarr_id TEXT,
    pin_verification TEXT NOT NULL CHECK (pin_verification IN ('published_version', 'local_checksum', 'api_postcheck')),
    refusal_reason TEXT,
    evidence_level TEXT CHECK (evidence_level IN ('spike_times', 'waveforms', 'raw')),
    sample_rate_hz REAL,
    analysis_sample_rate_hz REAL,
    duration_s REAL,
    n_channels INTEGER,
    author_label_source TEXT,
    unique_metadata_json TEXT,
    units_before INTEGER,
    good_before INTEGER,
    mua_before INTEGER,
    noise_before INTEGER,
    unlabeled_before INTEGER,
    units_after INTEGER,
    good_after INTEGER,
    mua_after INTEGER,
    noise_after INTEGER,
    n_merges INTEGER,
    n_splits INTEGER,
    n_removed INTEGER,
    n_relabelled INTEGER,
    n_rp_contaminated INTEGER,
    n_good_rp_contaminated INTEGER,
    n_rp_screened INTEGER,
    n_good_rp_screened INTEGER,
    n_isi_rate_flagged INTEGER,
    n_low_presence INTEGER,
    n_unstable INTEGER,
    n_amp_cutoff INTEGER,
    n_noise_as_good INTEGER,
    median_quality_before REAL,
    median_quality_after REAL,
    nwbi_critical INTEGER,
    nwbi_violation INTEGER,
    nwbi_suggestion INTEGER,
    PRIMARY KEY (review_id, asset_id),
    UNIQUE (review_id, path),
    FOREIGN KEY (review_id, dandiset_id) REFERENCES reviews(review_id, dandiset_id),
    FOREIGN KEY (dandiset_id, asset_id) REFERENCES assets(dandiset_id, asset_id)
);

CREATE TABLE findings (
    review_id TEXT NOT NULL REFERENCES reviews(review_id),
    id TEXT NOT NULL,
    code TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('critical', 'major', 'minor', 'info')),
    scope TEXT NOT NULL CHECK (scope IN ('dandiset', 'asset', 'unit')),
    statement_md TEXT NOT NULL,
    human_verified INTEGER NOT NULL CHECK (human_verified IN (0, 1)),
    asset_id TEXT,
    unit_ids_json TEXT,
    unit_id_namespace TEXT CHECK (unit_id_namespace IN ('nwb_unit_id', 'sorrel_cluster_index')),
    n_units INTEGER,
    metric TEXT,
    value REAL,
    threshold REAL,
    evidence_json TEXT,
    PRIMARY KEY (review_id, id),
    FOREIGN KEY (review_id, asset_id) REFERENCES review_assets(review_id, asset_id)
);

CREATE TABLE improvements (
    review_id TEXT NOT NULL REFERENCES reviews(review_id),
    id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('relabel', 'merge', 'split', 'remove', 'metadata_fix', 'nwb_fix', 'add_qc_columns')),
    addresses_json TEXT NOT NULL,
    description_md TEXT NOT NULL,
    asset_id TEXT,
    n_units INTEGER,
    metric TEXT,
    before REAL,
    after REAL,
    deliverable_sha256 TEXT CHECK (
        length(deliverable_sha256) = 64 AND deliverable_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (review_id, id),
    FOREIGN KEY (review_id, asset_id) REFERENCES review_assets(review_id, asset_id)
);

CREATE TABLE feedback (
    feedback_id TEXT PRIMARY KEY NOT NULL,
    dandiset_id TEXT NOT NULL REFERENCES dandisets(dandiset_id),
    review_ids_json TEXT NOT NULL,
    quoted_finding_ids_json TEXT NOT NULL,
    channel TEXT NOT NULL CHECK (channel IN ('email', 'github_issue', 'github_pr', 'dandi_discussion', 'other')),
    recipient TEXT,
    url TEXT,
    status TEXT NOT NULL CHECK (status IN ('drafted', 'sent', 'reminded', 'acknowledged', 'accepted', 'republished', 'declined', 'no_response_published')),
    disclosure_days INTEGER NOT NULL DEFAULT 30 CHECK (disclosure_days BETWEEN 14 AND 90),
    reminder_after_days INTEGER NOT NULL DEFAULT 14 CHECK (reminder_after_days >= 7),
    first_notified_at TEXT,
    first_notified_epoch INTEGER,
    reminder_due_epoch INTEGER,
    reminder_sent_at TEXT,
    reminder_sent_epoch INTEGER,
    publish_eligible_at TEXT,
    publish_eligible_epoch INTEGER,
    extension_until TEXT,
    extension_until_epoch INTEGER,
    extension_reason TEXT,
    published_at TEXT,
    published_epoch INTEGER,
    resulting_version TEXT,
    resulting_doi TEXT,
    units_corrected INTEGER CHECK (units_corrected >= 0),
    response_summary TEXT,
    CHECK (extension_until IS NULL OR extension_reason IS NOT NULL)
);

CREATE TABLE feedback_reviews (
    feedback_id TEXT NOT NULL REFERENCES feedback(feedback_id),
    review_id TEXT NOT NULL REFERENCES reviews(review_id),
    PRIMARY KEY (feedback_id, review_id)
);

CREATE TABLE feedback_events (
    feedback_id TEXT NOT NULL REFERENCES feedback(feedback_id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('drafted', 'sent', 'reminded', 'acknowledged', 'accepted', 'republished', 'declined', 'no_response_published')),
    actor TEXT NOT NULL,
    evidence_url TEXT,
    note TEXT,
    subject_json TEXT,
    recipient TEXT,
    channel TEXT CHECK (channel IN ('email', 'github_issue', 'github_pr', 'dandi_discussion', 'other')),
    early_followup INTEGER CHECK (early_followup = 1),
    PRIMARY KEY (feedback_id, ordinal)
);

CREATE TABLE review_artifacts (
    review_id TEXT NOT NULL REFERENCES reviews(review_id),
    path TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('report_md', 'cluster_qc_before', 'cluster_qc_after', 'clusters_before', 'clusters_after', 'describe_json', 'curation', 'report_html', 'nwb_inspector', 'dandiset_metadata', 'other')),
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    PRIMARY KEY (review_id, path)
);

CREATE VIEW v_latest_reviews AS
SELECT *
FROM reviews
WHERE review_id NOT IN (SELECT supersedes FROM reviews WHERE supersedes IS NOT NULL);

-- One row per (review, asset). Dandiset findings are reported separately.
-- PostgreSQL: replace group_concat with string_agg (same args).
CREATE VIEW v_result_table AS
WITH model_lists AS (
    SELECT DISTINCT review_id,
        group_concat(role || ':' || provider || '/' || model_id, ', ') OVER (
            PARTITION BY review_id ORDER BY ordinal
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS models
    FROM review_models
), finding_counts AS (
    SELECT review_id, asset_id, COUNT(*) AS n_findings, SUM(human_verified) AS n_verified_findings
    FROM findings GROUP BY review_id, asset_id
), review_feedback AS (
    -- Prefer the most recently notified record for the review, never another
    -- review's outreach or a newer unsent draft.
    SELECT fr.review_id, f.feedback_id, f.status, ROW_NUMBER() OVER (
        PARTITION BY fr.review_id
        ORDER BY f.status = 'drafted', f.first_notified_epoch DESC, f.feedback_id
    ) AS preference
    FROM feedback_reviews AS fr JOIN feedback AS f USING (feedback_id)
)
SELECT r.review_id, d.dandiset_id, r.version, r.draft_modified,
    d.name AS dandiset_name, d.doi AS dandiset_doi, d.license AS dandiset_license,
    r.created_at, r.mode, r.contributor, r.sorrel_version, r.sorrel_rev,
    r.skill_rev, r.prompt_rev, r.learned_rev, r.selection_policy, r.selection_scope, r.n_assets_total, r.n_nwb_assets,
    COALESCE(m.models, '') AS models, r.wall_s, a.asset_id,
    x.path, x.size_bytes, x.encoding, x.is_nwb, x.open_status, x.sha256, x.zarr_checksum, x.locator,
    x.media_type, x.download_url, x.blob_id, x.zarr_id, x.pin_verification,
    x.refusal_reason, x.evidence_level, x.sample_rate_hz, x.analysis_sample_rate_hz, x.duration_s, x.n_channels,
    x.author_label_source, x.units_before, x.good_before, x.mua_before, x.noise_before,
    x.unlabeled_before, x.units_after, x.good_after, x.mua_after, x.noise_after,
    x.n_merges, x.n_splits, x.n_removed, x.n_relabelled, x.n_rp_contaminated,
    x.n_good_rp_contaminated, x.n_rp_screened, x.n_good_rp_screened, x.n_isi_rate_flagged,
    x.n_low_presence, x.n_unstable, x.n_amp_cutoff,
    x.n_noise_as_good, x.median_quality_before, x.median_quality_after,
    x.nwbi_critical, x.nwbi_violation, x.nwbi_suggestion,
    COALESCE(f.n_findings, 0) AS n_findings,
    COALESCE(f.n_verified_findings, 0) AS n_verified_findings,
    COALESCE(g.n_findings, 0) AS dandiset_finding_count,
    COALESCE(g.n_verified_findings, 0) AS dandiset_verified_finding_count,
    fb.feedback_id, fb.status AS feedback_status
FROM v_latest_reviews AS r
JOIN dandisets AS d ON d.dandiset_id = r.dandiset_id
JOIN review_assets AS x ON x.review_id = r.review_id
JOIN assets AS a ON a.dandiset_id = d.dandiset_id AND a.asset_id = x.asset_id
LEFT JOIN model_lists AS m ON m.review_id = r.review_id
LEFT JOIN finding_counts AS f ON f.review_id = r.review_id AND f.asset_id = a.asset_id
LEFT JOIN finding_counts AS g ON g.review_id = r.review_id AND g.asset_id IS NULL
LEFT JOIN review_feedback AS fb ON fb.review_id = r.review_id AND fb.preference = 1;

-- Denominators require observed metrics at the asset level; missing unit
-- measurements within such an asset make the contamination share a floor.
-- Sum cost once per latest review and prefer a recorded total over its model subtotals.
CREATE VIEW v_scientific_impact AS
WITH model_costs AS (
    SELECT review_id, SUM(cost_usd) AS cost_usd FROM review_models GROUP BY review_id
), review_cost AS (
    SELECT SUM(COALESCE(r.cost_usd, m.cost_usd)) AS total_cost_usd,
        COUNT(COALESCE(r.cost_usd, m.cost_usd)) AS n_reviews_with_cost
    FROM v_latest_reviews AS r LEFT JOIN model_costs AS m ON r.review_id = m.review_id
), outreach AS (
    SELECT COUNT(first_notified_at) AS n_feedback_sent,
        COUNT(CASE WHEN status = 'republished' THEN 1 END) AS n_feedback_republished,
        SUM(CASE WHEN status = 'republished' THEN units_corrected END) AS units_corrected,
        COUNT(CASE WHEN status = 'republished' THEN units_corrected END) AS n_feedback_with_units_corrected
    FROM feedback
), assets_with_coverage AS (
    SELECT *,
        good_before > 0 AND n_good_rp_contaminated IS NOT NULL AS has_good_rp_contaminated,
        good_before > 0 AND n_good_rp_screened IS NOT NULL AS has_good_rp_screened
    FROM v_result_table
), asset_impact AS (
    SELECT COUNT(DISTINCT dandiset_id) AS dandisets_reviewed,
        COUNT(*) AS assets_reviewed,
        COUNT(CASE WHEN open_status = 'ok' THEN 1 END) AS assets_ok,
        COUNT(CASE WHEN open_status = 'refused' THEN 1 END) AS assets_refused,
        COUNT(units_before) AS n_assets_with_units,
        COUNT(n_rp_contaminated) AS n_assets_with_rp_post,
        COUNT(n_rp_screened) AS n_assets_with_rp_screen,
        COUNT(n_isi_rate_flagged) AS n_assets_with_isi_rate,
        SUM(units_before) AS units_before,
        SUM(good_before) AS good_before,
        SUM(n_rp_contaminated) AS n_rp_contaminated,
        SUM(n_rp_screened) AS n_rp_screened,
        SUM(n_isi_rate_flagged) AS n_isi_rate_flagged,
        SUM(n_good_rp_contaminated) AS n_good_rp_contaminated,
        SUM(n_good_rp_screened) AS n_good_rp_screened,
        COUNT(*) FILTER (WHERE has_good_rp_contaminated) AS n_assets_with_good_rp_contaminated,
        SUM(good_before) FILTER (WHERE has_good_rp_contaminated) AS good_before_with_rp_contaminated,
        100.0 * SUM(n_good_rp_contaminated) FILTER (WHERE has_good_rp_contaminated)
            / SUM(good_before) FILTER (WHERE has_good_rp_contaminated)
            AS pct_good_rp_contaminated,
        COUNT(*) FILTER (WHERE has_good_rp_screened) AS n_assets_with_good_rp_screened,
        SUM(good_before) FILTER (WHERE has_good_rp_screened) AS good_before_with_rp_screened,
        100.0 * SUM(n_good_rp_screened) FILTER (WHERE has_good_rp_screened)
            / SUM(good_before) FILTER (WHERE has_good_rp_screened)
            AS pct_good_rp_screened,
        100.0 * COUNT(CASE WHEN units_before > 0 AND unlabeled_before = units_before THEN 1 END)
            / NULLIF(COUNT(CASE WHEN units_before > 0 AND unlabeled_before IS NOT NULL THEN 1 END), 0)
            AS pct_assets_unlabeled,
        COUNT(COALESCE(n_merges, n_splits, n_removed, n_relabelled)) AS n_assets_with_curation,
        SUM(n_merges) AS n_merges,
        SUM(n_splits) AS n_splits,
        SUM(n_removed) AS n_removed,
        SUM(n_relabelled) AS n_relabelled
    FROM assets_with_coverage
)
SELECT * FROM asset_impact CROSS JOIN review_cost CROSS JOIN outreach;

-- PostgreSQL: use EXTRACT(EPOCH FROM date_trunc('second', CURRENT_TIMESTAMP))::BIGINT.
-- The clock CTE is also bound to an explicit timestamp by `impact --as-of`.
CREATE VIEW v_disclosure_queue AS
WITH clock(now) AS (SELECT CAST(strftime('%s', 'now') AS INTEGER))
SELECT f.*, CASE
    WHEN status IN ('republished', 'no_response_published') THEN 'published'
    WHEN status IN ('acknowledged', 'accepted') AND clock.now < extension_until_epoch THEN 'extended'
    WHEN status IN ('acknowledged', 'accepted', 'declined') THEN 'resolved'
    WHEN status IN ('sent', 'reminded') AND clock.now >= publish_eligible_epoch
        THEN 'publish_eligible_ghosted'
    WHEN status = 'sent' AND reminder_sent_epoch IS NULL AND clock.now >= reminder_due_epoch THEN 'reminder_due'
    ELSE 'in_grace_period'
END AS bucket
FROM feedback AS f CROSS JOIN clock;
