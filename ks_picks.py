def write_picks_to_sheet(gc: gspread.Client, sheet_id: str, picks: pd.DataFrame) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("Top_KS_Picks")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Top_KS_Picks", rows=100, cols=15)

    if "pitching_team" in picks.columns and "team" not in picks.columns:
        picks = picks.copy()
        picks["team"] = picks["pitching_team"]
    elif "pitcher_team" in picks.columns and "team" not in picks.columns:
        picks = picks.copy()
        picks["team"] = picks["pitcher_team"]

    output_cols = {
        "rank":             "Rank",
        "pitcher_name":     "Pitcher",
        "team":             "Team",
        "k_pct_season":     "K%",
        "swstr_pct":        "SwStr%",
        "chase_rate":       "Chase%",
        "avg_ip_per_start": "Avg IP/Start",
        "k_line":           "K Line",
        "ks_over_odds":     "Over Odds",
        "projected_k_calc": "Proj K",
        "prop_signal":      "Signal",
        "reason":           "Key Reasons",
        "confidence":       "Confidence",
    }

    available = {k: v for k, v in output_cols.items() if k in picks.columns}
    out_df    = picks[list(available.keys())].rename(columns=available)
    out_df    = out_df.copy().replace([np.inf, -np.inf], np.nan).fillna("")

    with_retry(lambda: ws.update([out_df.columns.tolist()] + out_df.astype(str).values.tolist()))

    ws_id  = ws.id
    n_cols = len(out_df.columns)
    reqs   = []

    # Base formatting — no wrap, clip
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": len(out_df) + 2, "startColumnIndex": 0, "endColumnIndex": n_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10},
            "verticalAlignment": "MIDDLE",
            "wrapStrategy": "CLIP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
    }})

    # Header row
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_HEADER,
            "textFormat": {"foregroundColor": COLOR_WHITE, "bold": True, "fontFamily": "Roboto", "fontSize": 11},
            "horizontalAlignment": "CENTER",
            "verticalAlignment": "MIDDLE",
            "wrapStrategy": "CLIP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
    }})

    # Alternating row colors
    for i in range(len(out_df)):
        bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": i + 1, "endRowIndex": i + 2, "startColumnIndex": 0, "endColumnIndex": n_cols},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat(backgroundColor)",
        }})

    # Column widths: Rank, Pitcher, Team, K%, SwStr%, Chase%, Avg IP/Start, K Line, Over Odds, Proj K, Signal, Key Reasons, Confidence
    col_widths = [45, 160, 55, 55, 65, 65, 90, 65, 90, 60, 150, 300, 90]
    for i, w in enumerate(col_widths[:n_cols]):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w},
            "fields": "pixelSize",
        }})

    # Row height — compact
    for i in range(len(out_df) + 1):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": 32},
            "fields": "pixelSize",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1}, "tabColorStyle": {"rgbColor": COLOR_HEADER}},
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
    except APIError as e:
        print(f"KS formatting failed: {e}")

    print(f"Written {len(out_df)} K picks to Top_KS_Picks")
