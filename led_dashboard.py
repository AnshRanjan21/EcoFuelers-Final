import streamlit as st
import pandas as pd
import httpx
from datetime import datetime, timedelta, timezone
import io
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import getpass
import os
import altair as alt
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")  

API_BASE = "http://localhost:8000/api"         # FastAPI base url
HISTORY_WINDOW_H = 24                          # hours of data to plot
REQUEST_TIMEOUT  = 15.0                        # seconds

COST_PER_MIN_WATT = 0.05         # ₹


# ---------- helper functions ----------------------------------------
def register_manual(led_id: int, when_utc: datetime) -> bool:
    try:
        r = httpx.post(
            f"{API_BASE}/register_manual",
            json={"led_id": led_id, "ts_utc": when_utc.isoformat()},
            timeout=REQUEST_TIMEOUT,
        )
        return r.status_code == 201
    except httpx.RequestError:
        return False

def switch_to_auto(led_id: int) -> bool:
    try:
        r = httpx.post(
            f"{API_BASE}/auto_brightness",
            json={"led_id": led_id},
            timeout=REQUEST_TIMEOUT,
        )
        return r.status_code == 200
    except httpx.RequestError:
        return False

@st.cache_data(ttl=3000)
def fetch_led_list() -> dict[int, str]:
    """Return {led_id: 'LED 3 – 18.75 W', …} for dropdown."""
    resp = httpx.get(f"{API_BASE}/leds", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return {row["id"]: f"LED {row['id']} – {row['wattage']} W" for row in data}

@st.cache_data(ttl=30)
def fetch_history_df(led_id: int, hours: int = HISTORY_WINDOW_H) -> pd.DataFrame:
    """Return a DataFrame indexed by ts with a 'level' column."""
    resp = httpx.get(
        f"{API_BASE}/leds/{led_id}/history",
        params={"hours": hours},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json()                         
    if not rows:
        return pd.DataFrame([], columns=["ts", "level"]).set_index("ts")

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts").sort_index()

def override_brightness(led_id: int, level: int) -> bool:
    """
    POST /api/override_brightness
    """
    try:
        r = httpx.post(
            f"{API_BASE}/override_brightness",
            json={"led_id": led_id, "level": level},
            timeout=REQUEST_TIMEOUT,
        )
        return r.status_code == 201      
    except httpx.RequestError:
        return False

@st.cache_data(ttl=300)
def fetch_brightness_df(led_id: int, hours: int) -> pd.DataFrame:
    resp = httpx.get(
        f"{API_BASE}/leds/{led_id}/history",
        params={"hours": hours},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json()

    if not rows:
        return pd.DataFrame([], columns=["ts", "level"]).set_index("ts")

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts").sort_index()

def compute_savings(
    df: pd.DataFrame,
    wattage: float,
    rate_per_min_watt: float,
    period_start: datetime,
    period_end: datetime,
) -> float:
    """
    Integrate (1‑level/100) × wattage × rate over the window.
    """
    if df.empty:
        return 0.0

    df = df.copy()  #  avoids SettingWithCopyWarning

    # Ensure the period start/end are covered
    if df.index[0] > period_start:
        df.loc[period_start] = df["level"].iloc[0]
    if df.index[-1] < period_end:
        df.loc[period_end] = df["level"].iloc[-1]

    df = df.sort_index()

    # Duration until next reading
    next_idx = df.index[1:].tolist() + [period_end]
    minutes  = (pd.Series(next_idx, index=df.index)
                  .subtract(df.index)
                  .dt.total_seconds() / 60.0)

    savings  = rate_per_min_watt * wattage * minutes * (1 - df["level"] / 100)
    return savings.sum()

# ---------- helper: sensor history -----------------------------------------
@st.cache_data(ttl=300)
def fetch_sensor_df(led_id: int, hours: int) -> pd.DataFrame:
    resp = httpx.get(
        f"{API_BASE}/leds/{led_id}/sensor_history",
        params={"hours": hours},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return pd.DataFrame([], columns=["ts", "lux"]).set_index("ts")

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts").sort_index()

# ---------- LLM helper ------------------------------------------------------
def get_groq_model():
    """
    Lazily initialise the Groq Llama‑3 chat model.
    Will prompt for the API key once per session if the env var is missing.
    """
    import getpass, os
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter Groq API key: ")

    from langchain.chat_models import init_chat_model
    return init_chat_model("llama3-8b-8192", model_provider="groq")


# ---------- helper: prep a concise prompt -----------------------------------
def build_schedule_prompt(clean_df: pd.DataFrame, led_label: str) -> list[dict]:
    """
    Build a prompt for an hourly (07:00–02:00) brightness schedule based on sensor_lux.
    The LLM must return a Markdown table only, with two columns: time | brightness %.
    """
    # Use at most the first 200 rows for context
    csv_snippet = clean_df.head(200).to_csv(index=True)

    return [
        {
            "role": "system",
            "content": (
                "You are an energy‑efficiency consultant specialising in adaptive LED lighting.\n\n"
                "### Your task\n"
                "From the past 48 h of **(timestamp,sensor_lux)** data provided, "
                "produce an **hourly schedule** for this fixture starting at **07:00** and ending at **02:00** next day "
                "(20 rows in total).\n\n"
                "Return a **Markdown table only**—no prose—with **two columns**:\n"
                "1. `time` (07:00, 08:00, …, 02:00)\n"
                "2. `brightness %` (nearest 5 %)\n\n"
                "### Brightness formula\n"
                "```text\n"
                "brightness% = round_to_nearest_5( max(40, 100 − sensor_lux / 10) )\n"
                "```\n"
                "- Clamp brightness to the range **40 %–100 %**.\n"
                "- Always round to the nearest **5 %**.\n\n"
                "**Important:** Output the table only—no additional explanation or code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Fixture: {led_label}\n\n"
                "Here are the last 48 hours of readings (ISO_TIMESTAMP,sensor_lux):\n\n"
                f"{csv_snippet}\n\n"
                "Generate the hourly schedule following the rules above."
            ),
        },
    ]


# --------------------------------------------------------------------
# Page renderers
# --------------------------------------------------------------------

def render_home() -> None:
    from datetime import datetime

    st.title("💡 LED Brightness – 24 h Trend")

    # 1️⃣ Fixture selector
    led_lookup = fetch_led_list()
    if not led_lookup:
        st.error("No LEDs returned by the API.")
        return

    selected_id = st.selectbox(
        "Select a fixture",
        options=list(led_lookup.keys()),
        format_func=led_lookup.get,
    )

    # 2️⃣ Fetch history
    hist_df = fetch_history_df(selected_id)
    if not hist_df.empty and hist_df.index.tz is None:
        hist_df.index = hist_df.index.tz_localize("UTC")
    if not hist_df.empty:
        hist_df.index = hist_df.index.tz_convert(IST)

    latest_lv = int(hist_df["level"].iloc[-1]) if not hist_df.empty else None

    # ── session state for mode ─────────────────────────────────────────
    mode_key = f"manual_mode_{selected_id}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = False  # False → Auto

    # 3️⃣ KPIs with placeholders
    col_bri, col_mode = st.columns(2)
    bri_ph  = col_bri.empty()
    mode_ph = col_mode.empty()

    # initial render
    bri_ph.metric("Current brightness", f"{latest_lv} %" if latest_lv is not None else "—")
    mode_ph.metric("Mode", "Manual" if st.session_state[mode_key] else "Auto")

    # 4️⃣ Manual override controls
    st.markdown("#### Manual override")
    new_level = st.slider(
        "Brightness (%)",
        0, 100, latest_lv or 50, 1,
        key="override_slider",
        label_visibility="collapsed",
    )

    col_apply, col_auto = st.columns(2)

    with col_apply:
        if st.button("Apply", key="override_apply"):
            if override_brightness(selected_id, new_level):
                register_manual(selected_id, datetime.utcnow())
                st.session_state[mode_key] = True     # now Manual

                st.success(f"Brightness set to {new_level}%")
                # update placeholders (no duplicate rows)
                bri_ph.metric("Current brightness", f"{new_level} %")
                mode_ph.metric("Mode", "Manual")

                # optimistic chart update
                if not hist_df.empty:
                    hist_df.loc[datetime.now(IST)] = new_level
            else:
                st.error("Failed to apply override")

    with col_auto:
        if st.button("Back to auto", key="auto_mode"):
            if switch_to_auto(selected_id):
                st.session_state[mode_key] = False    # back to Auto
                st.success("Switched to auto‑brightness")
                # update placeholders
                mode_ph.metric("Mode", "Auto")
            else:
                st.error("Failed to switch to auto")

    # 5️⃣ Chart
    if hist_df.empty:
        st.info("No brightness records for the past 24 hours.")
    else:
        st.line_chart(hist_df["level"])

    # 6️⃣ Footer
    st.caption(
        f"Data window: last {HISTORY_WINDOW_H} h · "
        f"Updated {datetime.now(IST):%Y‑%m‑%d %H:%M IST}"
    )


def render_cost_savings() -> None:
    """
    Live cost‑savings analytics page.
    Shows ₹ saved per LED and headline KPIs for the last 24 h and 30 d,
    plus an explanation of the maths.
    """
    st.title("💰 Cost Savings Overview")

    # 🔧— CSS: turn KPI text green
    st.markdown(
        """
        <style>
        /* metric header + value */
        div[data-testid="metric-container"] > label,
        div[data-testid="metric-container"] > div {
            color: #2e8b57 !important;   /* sea‑green */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1️⃣ Fetch LED inventory
    led_info = fetch_led_list()
    led_ids  = list(led_info.keys())

    # 2️⃣ Time windows
    now, start_24h, start_30d = datetime.utcnow(), datetime.utcnow() - timedelta(hours=24), datetime.utcnow() - timedelta(days=30)

    # 3️⃣ Aggregate savings
    rows, total_24h, total_30d = [], 0.0, 0.0
    for led_id in led_ids:
        wattage = float(led_info[led_id].split("–")[1].split()[0])
        df_30d  = fetch_brightness_df(led_id, hours=720)

        saved_24h = compute_savings(df_30d[df_30d.index >= start_24h], wattage, COST_PER_MIN_WATT, start_24h, now)
        saved_30d = compute_savings(df_30d, wattage, COST_PER_MIN_WATT, start_30d, now)

        rows.append({"LED": led_id, "₹ Saved (24 h)": saved_24h, "₹ Saved (30 d)": saved_30d})
        total_24h += saved_24h
        total_30d += saved_30d

    # 4️⃣ Headline KPIs (💰 + green)
    col1, col2 = st.columns(2)
    col1.metric("💰 Saved last 24 h", f"₹ {total_24h:,.0f}")
    col2.metric("💰 Saved last 30 d", f"₹ {total_30d:,.0f}")

    # 5️⃣ Per‑fixture table
    df_leds = pd.DataFrame(rows).set_index("LED").round(0).astype(int)
    st.markdown("### Per‑fixture detail")
    st.dataframe(df_leds, use_container_width=True)

    # 6️⃣ Green bar chart (Altair)
    chart = (
        alt.Chart(df_leds.reset_index())
        .mark_bar(color="#2e8b57")          # same sea‑green
        .encode(
            x=alt.X("LED:N", title="LED"),
            y=alt.Y("₹ Saved (30 d):Q", title="₹ Saved (30 d)"),
            tooltip=["LED", "₹ Saved (30 d)"]
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # 7️⃣ Explain the maths
    st.markdown("#### How we compute these savings")
    st.markdown(
        """
        1. The numbers above represent **how much money you saved compared with keeping every
           light at 100 % brightness for the whole period**.
        2. We assume power draw scales linearly with brightness, so dimming directly
           reduces energy consumed – and therefore your bill.
        """
    )
    with st.expander("Show exact formula"):
        st.code(
            r"""
₹_saved = Σ_minutes [ wattage_W × (1 − brightness_pct / 100) × COST_PER_MIN_WATT ]
""",
            language="text",
        )
        st.markdown(
            f"where **COST_PER_MIN_WATT = ₹ {COST_PER_MIN_WATT:.6f}** (rupees per watt‑minute)."
        )

def render_energy_plan() -> None:
    """Suggest an energy‑saving plan via Groq Llama‑3."""
    st.title("🧠 Suggest energy‑saver plan")

    # ╭─ 0️⃣  Quick instructions ───────────────────────────────────────────╮
    st.info(
        "### How to use\n"
        "1. **Select a fixture** *or* **upload a CSV** you exported earlier.\n"
        "2. Click **AI scheduler** to generate a 24‑hour energy‑saving plan.\n"
        "   · The app looks at the last 48 h of data (or the file you supply).\n"
        "   · The plan appears below in a few seconds.\n"
    )

    # ╭─ 1️  LED chooser (always visible) ─────────────────────────────────╮
    led_lookup = fetch_led_list()
    led_id = st.selectbox(
        "Select a fixture",
        options=[None] + list(led_lookup.keys()),
        format_func=lambda x: "—" if x is None else led_lookup[x],
        index=0,
    )

    # ╭─ 2️  Optional CSV upload ─────────────────────────────────────────╮
    csv_file = st.file_uploader("…or upload a CSV (optional)", type="csv")

    # ╭─ 3️  AI scheduler button ─────────────────────────────────────────╮
    run_ai = st.button(
        "⚡ AI scheduler",
        disabled=(led_id is None and csv_file is None),
    )

    # ╭─ 4️ When pressed, gather data & call Groq Llama‑3 ───────────────╮
    if run_ai:
        with st.spinner("Talking to Llama‑3…"):
            # ❶  Build dataframe
            if csv_file:
                df = pd.read_csv(csv_file, parse_dates=["ts"]).set_index("ts")
                led_label = csv_file.name.replace(".csv", "")
            else:
                df = fetch_sensor_df(led_id, 48)     
                df = df.rename(columns={"lux": "sensor_lux"})
                led_label = led_lookup[led_id]


            if df.empty:
                st.error("No data available for the selected source.")
                return

            if df.index.tz is None:              # localise naïve index to UTC first
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(IST)  # finally convert to IST

            # ❷  Send to LLM
            messages = build_schedule_prompt(df, led_label)
            model    = get_groq_model()
            answer   = model.invoke(messages).content

        # ❸  Show results
        st.subheader("Proposed plan")
        st.markdown(answer)
        with st.expander("📈 Data preview (top 100 rows)"):
            st.dataframe(df.head(100))

def render_data_export() -> None:
    """Interactive extractor for brightness or sensor data."""
    st.title("📤 Data Export")

    # 1️⃣ LED picker
    led_lookup = fetch_led_list()
    if not led_lookup:
        st.error("No LEDs returned by the API.")
        return

    led_id = st.selectbox(
        "Select a fixture",
        options=list(led_lookup.keys()),
        format_func=led_lookup.get,
    )

    # 2️⃣ Window & data‑type selectors
    col_a, col_b = st.columns(2)
    with col_a:
        window_choice = st.radio(
            "Time window",
            ["Past 7 days", "Past 30 days"],
            horizontal=True,
        )
        hours = 168 if window_choice.startswith("Past 7") else 720

    with col_b:
        dtype = st.radio(
            "Data type",
            ["Brightness level (%)", "Sensor lux value"],
            horizontal=True,
        )

    # 3️⃣ Fetch data
    if dtype.startswith("Brightness"):
        df = fetch_brightness_df(led_id, hours)
        df = df.rename(columns={"level": "brightness"})
        filename = f"led{led_id}_brightness.csv"
    else:
        df = fetch_sensor_df(led_id, hours)
        df = df.rename(columns={"lux": "sensor_lux"})
        filename = f"led{led_id}_sensor.csv"

    # 4️⃣ Display + download
    if df.empty:
        st.info("No records for the selected range.")
        return

    st.dataframe(df)

    csv_bytes = df.to_csv().encode()
    st.download_button(
        label="💾 Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

# --------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Adaptive Lighting Dashboard", layout="centered")
    
    # Sidebar navigation
    st.sidebar.title("🔀 Navigation")
    page = st.sidebar.radio("Jump to:", ["Home", "Cost savings", "Smart AI planning", "Export data"], index=0)

    # Route to selected page
    if page == "Home":
        render_home()
    elif page == "Cost savings":
        render_cost_savings()
    elif page == "Export data":
        render_data_export()
    elif page == "Smart AI planning":
        render_energy_plan()

# Run immediately when Streamlit executes the file
if __name__ == "__main__":
    main()
